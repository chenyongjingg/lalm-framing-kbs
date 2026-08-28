#!/usr/bin/env python3
"""
Standalone E2B text-only P1-PILOT inference on GPU 1.
Runs in parallel with main P1-PILOT (E4B on GPU 0).
Writes to same checkpoint/response format so main process will skip E2B when it finishes E4B.
SAFE: No modification to running code. Independent process. Separate log.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Force GPU 1

import sys, json, time, argparse
from pathlib import Path
import torch

# Add pipeline dir to path
sys.path.insert(0, "/root/lalm_framing_revision_v6")

from common_utils import load_config, setup_logging, JsonlLogger, Checkpoint, ModelManager
from stage_p1_pilot import build_design, STAGE

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.yaml")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["workdir"]).expanduser()

    # Use separate log file to avoid conflict with main process
    log = setup_logging(str(root / "logs" / "p1_pilot_e2b_gpu1.log"), "p1_pilot_e2b_gpu1")
    log.info("=== E2B GPU-1 Standalone P1-PILOT Text Inference ===")
    log.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))
    log.info("GPU count visible: %d", torch.cuda.device_count())

    # Load frozen queries (same file as main process)
    full_pilot_f = root / "results" / "p1_pilot_queries_full.json"
    if not full_pilot_f.exists():
        log.error("Frozen PILOT queries not found: %s", full_pilot_f)
        return 3

    _fp = json.loads(full_pilot_f.read_text(encoding="utf-8"))
    queries = _fp.get("queries", [])
    n_q = 150  # PILOT quota
    queries = queries[:n_q]
    log.info("Loaded %d frozen PILOT queries", len(queries))

    # Build design (same as main process)
    p1p = cfg.get("p1_pilot", {})
    design = build_design(queries, p1p)
    log.info("Design cells: %d total", len(design))

    # Filter to E2B text-only cells
    e2b_cells = [c for c in design if c["A_s"] == "text"]
    log.info("E2B text-only cells: %d", len(e2b_cells))

    # Check if already done
    ckpt_e2b = Checkpoint(str(root / "checkpoints" / f"{STAGE}_gemma_4_e2b.jsonl"))
    resp_file = root / "responses" / "P1_PILOT" / "gemma_4_e2b_responses.jsonl"
    resp_file.parent.mkdir(parents=True, exist_ok=True)

    # Collect already-done response IDs
    resp_ids_done = set()
    if resp_file.exists():
        with open(resp_file, encoding="utf-8") as _rf:
            for _line in _rf:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    resp_ids_done.add(json.loads(_line).get("response_id", ""))
                except Exception:
                    continue

    pending = [c for c in e2b_cells
               if f"P1P_gemma_4_e2b_{c['query_id']}_{c['combo']}_{c['A_s']}_t{c['template_idx']}"
               not in resp_ids_done]

    log.info("Already done: %d, Pending: %d", len(resp_ids_done), len(pending))

    if not pending:
        log.info("E2B already complete. Exiting.")
        return 0

    # Load E2B model
    cfg_models = cfg.get("models", {})
    if "gemma_4_e2b" not in cfg_models:
        log.error("gemma_4_e2b not in config models")
        return 3

    mconf = cfg_models["gemma_4_e2b"]
    log.info("Loading Gemma-4-E2B-it on GPU 1...")
    mm = ModelManager(cfg, root)
    try:
        model, tok = mm.load("gemma_4_e2b", mconf)
        log.info("E2B model loaded. VRAM: %.1f GB", torch.cuda.memory_allocated(0) / 1e9)
    except Exception as e:
        log.error("E2B model load failed: %s", str(e)[:200])
        return 3

    # System prompt for Gemma-4
    sys_msg = mconf.get("system_prompt",
        "You are a careful, consistent assistant.\n<start_of_thinking>\n<enable_thinking>false</enable_thinking>\n<end_of_thinking>")
    sys_msg = sys_msg.strip()

    # E2B is text-only, use chat template
    max_new = args.max_new_tokens
    bs = args.batch_size
    done_count = 0
    t_start = time.time()

    with open(resp_file, "a", encoding="utf-8") as f:
        for i, cell in enumerate(pending):
            try:
                prompt_text = cell["template"].replace("{query}", cell["query_text"])
                msgs = [{"role": "system", "content": sys_msg},
                        {"role": "user", "content": prompt_text}]
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = tok(text=text, return_tensors="pt", truncation=True, max_length=4096)
                inputs = {k: v.to(model.device) if hasattr(v, "to") else v
                          for k, v in inputs.items()}

                with torch.no_grad():
                    o = model.generate(**inputs, max_new_tokens=max_new,
                                       do_sample=False, temperature=1.0,
                                       pad_token_id=tok.pad_token_id or tok.eos_token_id)
                response = tok.decode(o[0][inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True).strip()

                entry = {
                    "response_id": f"P1P_gemma_4_e2b_{cell['query_id']}_{cell['combo']}_{cell['A_s']}_t{cell['template_idx']}",
                    "model": "gemma_4_e2b",
                    "modality": "text",
                    "query_id": cell["query_id"],
                    "combo": str(cell["combo"]),
                    "A_s": cell["A_s"],
                    "template_idx": cell["template_idx"],
                    "query_text": cell["query_text"][:200],
                    "prompt": prompt_text[:500],
                    "response": response,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
                ckpt_e2b.mark_done(cell["query_id"], str(cell["combo"]),
                                   cell["A_s"], cell["template_idx"])
                done_count += 1

                if (i + 1) % 50 == 0:
                    elapsed = time.time() - t_start
                    rate = (i + 1) / elapsed * 3600
                    eta = (len(pending) - i - 1) / rate if rate > 0 else 0
                    log.info("[E2B GPU-1] Progress: %d/%d (%.1f%%), rate=%.1f/h, ETA=%.1fh, VRAM=%.1fGB",
                             i + 1, len(pending), 100*(i+1)/len(pending),
                             rate, eta, torch.cuda.memory_allocated(0) / 1e9)

            except Exception as e:
                log.warning("[E2B GPU-1] cell %d failed: %s", i, str(e)[:150])
                continue

    elapsed = time.time() - t_start
    log.info("[E2B GPU-1] COMPLETE: %d/%d in %.1fh (rate=%.1f/h)",
             done_count, len(pending), elapsed / 3600,
             done_count / elapsed * 3600 if elapsed > 0 else 0)
    return 0

if __name__ == "__main__":
    sys.exit(main())
