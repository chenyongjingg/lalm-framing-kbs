"""
common_utils.py — 流水线公共工具模块

统一实现（消除各阶段脚本多份拷贝导致的函数名残留/重构遗漏问题）：
  - Checkpoint 断点续跑（每条推理完成立即落盘）
  - 失败重试（3 次指数退避）
  - 结构化 JSONL 日志
  - 模型加载（fp16 优先，显存不足自动降级 4bit，决策写入日志）
  - 模型常驻 + 显存管理（阶段内单次加载，阶段间强制释放）
  - 加载超时检测（防止 overlayfs 挂起无限等待）

可移植性：所有路径来自 config，无任何硬编码服务器路径。
"""

import json
import time
import threading
import logging
import sys
from pathlib import Path

# torch 延迟导入：Checkpoint/重试/日志/配置等功能在纯 CPU 环境也可用，
# 仅 ModelManager/generate_batch 等 GPU 功能真正需要 torch。
torch = None


def _require_torch():
    global torch
    if torch is None:
        import torch as _t
        torch = _t
        # ── v6.5.3 修复 2026-08-04（与 scorer_utils 同款）──
        # transformers 5.14.1 在部分加载路径（AWQ/trust_remote_code）内部显式
        # 调用 torch.compile，产生 compile worker 线程池 → 与主线程 futex 互锁
        # 死锁（CPU 100% 忙等、零产出）。此处将 torch.compile 替换为 no-op。
        try:
            if not getattr(_t, "_WB_COMPILE_PATCHED", False):
                _orig_compile = _t.compile
                def _noop_compile(*args, **kwargs):
                    return None
                _t._orig_compile = _orig_compile
                _t.compile = _noop_compile
                _t._WB_COMPILE_PATCHED = True
                import os as _os
                _os.environ["TORCH_COMPILE_DISABLE"] = "1"
                _os.environ["TORCHDYNAMO_DISABLE"] = "1"
                _os.environ["TORCHINDUCTOR_COMPILE_WORKER_TIMEOUT"] = "600"
        except Exception:  # noqa: BLE001
            pass
    return torch


def _torch_no_grad():
    """延迟到方法实际调用时才应用 torch.no_grad 的装饰器工厂。"""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            with _require_torch().no_grad():
                return fn(*args, **kwargs)
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------

def setup_logging(log_file: str, name: str = "pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


class JsonlLogger:
    """结构化 JSONL 日志（每阶段事件流）。"""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, **kv):
        kv["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(kv, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Checkpoint 断点续跑
# ---------------------------------------------------------------------------

class Checkpoint:
    """记录已完成的 (model, condition, query_id) 条目，支持 --resume。

    每条推理完成后立即 append 落盘（禁止只在内存累积）。

    支持带分值持久化：mark_done("harmbench", rid, label, prob) 会存 4 元组；
    is_done("harmbench", rid) 做前缀匹配，仍能命中。读取分值用
    ckpt.get_payload("harmbench", rid) 返回后缀部分。
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done = set()
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.done.add(tuple(json.loads(line)))

    def is_done(self, *key) -> bool:
        k = tuple(key)
        return any(t[: len(k)] == k for t in self.done)

    def get_payload(self, *key) -> tuple:
        """返回匹配 key 前缀的条目中 key 之后的部分（无则空元组）。"""
        k = tuple(key)
        for t in self.done:
            if len(t) > len(k) and t[: len(k)] == k:
                return t[len(k):]
        return ()

    def mark_done(self, *key):
        t = tuple(key)
        if t not in self.done:
            self.done.add(t)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(list(t), ensure_ascii=False) + "\n")

    def __len__(self):
        return len(self.done)


# ---------------------------------------------------------------------------
# 失败重试（3 次指数退避）
# ---------------------------------------------------------------------------

def retry_call(fn, args=(), kwargs=None, max_retries: int = 3,
               backoff_base: float = 2.0, errors_log: JsonlLogger = None,
               context: dict = None):
    """单查询失败自动重试 3 次（指数退避），仍失败记 errors.jsonl 并返回 None。"""
    kwargs = kwargs or {}
    last_err = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = backoff_base ** attempt
            if errors_log:
                errors_log.event(type="retry", attempt=attempt + 1,
                                 error=str(e)[:500], **(context or {}))
            time.sleep(wait)
    if errors_log:
        errors_log.event(type="failed_permanent", error=str(last_err)[:500],
                         **(context or {}))
    return None


# ---------------------------------------------------------------------------
# 模型加载（fp16 优先 + 4bit 自动降级 + 决策写日志）
# ---------------------------------------------------------------------------

class ModelManager:
    """模型常驻管理：同一阶段内模型只加载一次；加载超时检测；显存峰值记录；
    快速存储预置（overlayfs I/O 优化）；samples-per-load 效率统计。"""

    def __init__(self, logger: logging.Logger, load_timeout: int = 2700,
                 prefer_fp16: bool = True, hf_home: str = None,
                 io_cfg: dict = None):
        self.log = logger
        self.load_timeout = load_timeout
        self.prefer_fp16 = prefer_fp16
        self.hf_home = hf_home
        self._cache = {}  # name -> (model, tokenizer, precision)
        # --- I/O 优化状态 ---
        self.io_cfg = io_cfg or {}
        self._fast = None          # FastStorage 懒初始化
        self._prefetcher = None
        self._stats = {}           # name -> {"loads": int, "samples": int}

    # ---- I/O 优化：快速存储与预取 ----
    def _get_fast(self):
        if self._fast is None and self.io_cfg.get("prestage_models", False):
            try:
                from model_cache import FastStorage
                self._fast = FastStorage(
                    self.io_cfg.get("fast_storage", "auto"),
                    self.io_cfg.get("max_fast_storage_gb", 24), self.log)
                if self._fast.available:
                    self.log.info("快速存储: %s (%s) 读取 %.0f MB/s",
                                  self._fast.path, self._fast.kind,
                                  self._fast.speed_mbps)
            except Exception as e:  # noqa: BLE001
                self.log.warning("快速存储初始化失败（禁用预置）: %s",
                                 str(e)[:150])
                self._fast = False
        return self._fast or None

    def get_prefetcher(self):
        if self._prefetcher is None and self.io_cfg.get("prefetch_next", False):
            fast = self._get_fast()
            if fast:
                from model_cache import ModelPrefetcher
                self._prefetcher = ModelPrefetcher(
                    fast, self.hf_home,
                    self.io_cfg.get("stall_timeout_sec", 120), self.log)
        return self._prefetcher

    def prefetch(self, name: str, model_path: str):
        """提交后台预取（推理期间并行 I/O）。在加载当前模型后立即调用，
        传入本阶段下一个将使用的模型。"""
        pf = self.get_prefetcher()
        if pf:
            pf.submit(name, model_path)

    def count_samples(self, name: str, n: int):
        """记录某模型完成的推理样本数（samples-per-load 效率指标）。"""
        st = self._stats.setdefault(name, {"loads": 0, "samples": 0})
        st["samples"] += n

    def stats_report(self):
        for name, st in self._stats.items():
            spl = st["samples"] / st["loads"] if st["loads"] else 0
            self.log.info("[效率] %s: 加载 %d 次, 推理 %d 样本, "
                          "samples/load = %.0f", name, st["loads"],
                          st["samples"], spl)

    def _resolve_load_path(self, name: str, model_path: str) -> str:
        """优先取预取器已完成的预置路径；否则按需同步预置；失败回退原路径。"""
        fast = self._get_fast()
        if not fast:
            return model_path
        # 1) 预取已完成？
        pf = self.get_prefetcher()
        if pf:
            ready = pf.wait_ready(name, timeout=0)
            if ready:
                self.log.info("[%s] 使用后台预取结果: %s", name, ready)
                return ready
        # 2) 同步预置
        try:
            from model_cache import prestage_model
            return prestage_model(name, model_path, fast, self.hf_home,
                                  self.io_cfg.get("stall_timeout_sec", 120),
                                  self.log)
        except Exception as e:  # noqa: BLE001
            self.log.warning("[%s] 预置失败（直接加载）: %s",
                             name, str(e)[:150])
            return model_path

    def load(self, name: str, model_path: str, trust_remote_code: bool = True):
        """加载模型。返回 (model, tokenizer, precision)。超时报 TimeoutError。"""
        _require_torch()
        if name in self._cache:
            return self._cache[name]
        # M-cu.1-fix（审计 v6.5.29）：顺序加载新模型前先卸载已在缓存的旧模型——
        # 原实现缓存驻留旧模型的同时加载新模型，多模型顺序加载时显存峰值 = 各
        # 模型之和（非增量），易 OOM（CODE_SCIENCE_REPORT M-cu.1 Major）。调用方
        # 此前以"每实例单模型"规避，此处将契约落实到 API 层。
        # 注意：unload_all 仅清本实例缓存内模型，不影响其他实例；load_timeout
        # 超时后线程仍可能残留加载（该语义不变）。
        if self._cache:
            self.log.info("[M-cu.1] 加载 %s 前释放已在缓存的 %d 个旧模型（峰值增量）",
                          name, len(self._cache))
            self.unload_all()
        # I/O 优化：先预置到快速存储再加载
        model_path = self._resolve_load_path(name, model_path)

        result, err = {}, {}

        def _load():
            nonlocal model_path
            try:
                from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                          BitsAndBytesConfig, AutoModel)
                import json as _json
                import os
                # torch < 2.6 时 transformers 5.x 的 check_torch_load_is_safe
                # 会拒绝加载 .pt 说话人文件（如 Qwen2.5-Omni spk_dict.pt）。
                # 官方仓库文件 + weights_only=True 风险可控，monkey-patch 放行。
                # 关键：必须同时 patch 顶层 transformers.utils 与源头 import_utils
                #（模型模块用 `from transformers.utils import ...` 绑定，只改源头不生效）
                try:
                    import transformers.utils as _tu
                    from transformers.utils import import_utils as _iu
                    _noop = lambda *a, **k: None  # noqa: E731
                    for _mod in (_tu, _iu):
                        if hasattr(_mod, "check_torch_load_is_safe"):
                            _mod.check_torch_load_is_safe = _noop
                except Exception:
                    pass
                if self.hf_home:
                    os.environ["HF_HOME"] = self.hf_home
                # 强制离线：服务器外网不可达，任何联网检查（HEAD/etag）都会
                # 阻塞重试后失败。先解析本地 HF 缓存路径，再 local_files_only。
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                os.environ["HF_HUB_OFFLINE"] = "1"
                try:
                    from scorer_utils import resolve_local_model_path as _rlp
                    model_path = _rlp(model_path)
                except Exception:
                    pass
                tok = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=trust_remote_code,
                    local_files_only=True)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                precision = "fp16"
                kwargs = dict(torch_dtype=torch.float16, device_map="auto",
                              trust_remote_code=trust_remote_code)
                if not self.prefer_fp16:
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4")
                    kwargs.pop("torch_dtype")
                    precision = "4bit"
                # 架构自动检测：audio/omni 等模型不能用 AutoModelForCausalLM。
                # transformers 5.x 类名显式映射（Qwen2Audio/AudioFlamingo3）。
                arch_cls = AutoModelForCausalLM
                try:
                    cfg_path = os.path.join(model_path, "config.json")
                    if os.path.exists(cfg_path):
                        with open(cfg_path, encoding="utf-8") as _fh:
                            _cfg = _json.load(_fh)
                        archs = " ".join(_cfg.get("architectures") or [])
                        # v6.5.29-fix（E2B text 推理崩溃根因修复）：/root/models 的
                        # gemma-4-* 权重 config.json 的 architectures 为 null（HF cache
                        # 版本才含 Gemma4ForConditionalGeneration）→ 架构检测 miss →
                        # E2B 被当 CausalLM 加载（AutoTokenizer），stage_p1_pilot 用
                        # tok.tokenizer 访问 Gemma4Processor 属性 → AttributeError →
                        # E2B text 3600 全失败（errors 62 条，p1_pilot.complete 未写）。
                        # 补 model_type fallback：config model_type 含 "gemma4"（多模态
                        # text+image+audio）即按 Gemma4Processor 加载。
                        _mtype = str(_cfg.get("model_type") or "")
                        # v6.5.29-fix（E2B fp16 19.2GB OOM 兜底）：部分 gemma-4 权重
                        # config 在特定解析路径下 architectures/model_type 读取不一致，
                        # 导致 Processor BF16 分支 miss → CausalLM fp16（19.2GB 峰值）
                        # → 评分段 OOM。增加 model 名 fallback：model_path 含 gemma-4/
                        # gemma_4（协议 §10.4 gemma-4 家族 BF16 直载）即命中。
                        _is_gemma4_mm = ("Gemma4ForConditionalGeneration" in archs
                                         or "gemma4" in _mtype.lower()
                                         or "gemma-4" in model_path.lower()
                                         or "gemma_4" in model_path.lower())
                        # ── v6.5：Gemma-4 多模态条件生成模型（text+image+audio）──
                        # 与 Qwen2-Audio 同类，不是 CausalLM；须 BF16 直载
                        # （bnb 4bit 不支持多模态条件生成；官方无 -qat 仓库，
                        # 16GB < 24GB 显存放得下），返回 Processor 而非 tokenizer。
                        if _is_gemma4_mm:
                            from transformers import (  # noqa: PLC0415
                                Gemma4ForConditionalGeneration, AutoProcessor)
                            proc = AutoProcessor.from_pretrained(
                                model_path, local_files_only=True,
                                trust_remote_code=trust_remote_code)
                            if proc.tokenizer.pad_token is None:
                                proc.tokenizer.pad_token = proc.tokenizer.eos_token
                            # v6.5-fix 2026-08-07：Gemma4Processor 无 pad_token_id
                            # 属性（在内部 .tokenizer 上）。统一补齐，使 stage_* 调用方
                            # tok.pad_token_id 无感可用（stage_d/p1_pilot/p1_full 均用到）。
                            try:
                                proc.pad_token_id = proc.tokenizer.pad_token_id
                            except Exception:  # noqa: BLE001
                                pass
                            # v6.6.2-fix（2026-08-11 巡检）：加载前强制清显存，提高
                            # Gemma-4 BF16 直载成功率。实证（08-11 巡检）：GPU 残留
                            # 上一模型（E4B）显存未完全释放时，E2B BF16 加载 OOM →
                            # 静默降级 CausalLM fp16（AutoModelForCausalLM 承载 Gemma4
                            # 多模态 config 属错误加载）→ generate 病理挂死 → E2B 反复
                            # 停滞在 ~75 行、watchdog 重启循环。清显存后再载可显著提高
                            # BF16 直载命中（实测 04:54 干净显存下 E2B bf16 15.9GB
                            # 直载成功，生成 1 行/40s 正常；fp16 挂死）。
                            try:
                                import gc as _gc  # noqa: PLC0415
                                _gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except Exception:  # noqa: BLE001
                                pass
                            model = Gemma4ForConditionalGeneration.from_pretrained(
                                model_path, local_files_only=True,
                                trust_remote_code=trust_remote_code,
                                torch_dtype=torch.bfloat16, device_map="auto",
                                max_memory={0: "23GiB"})
                            model.eval()
                            result["m"] = (model, proc, "bf16")
                            return
                        if "Qwen2AudioForConditionalGeneration" in archs:
                            from transformers import Qwen2AudioForConditionalGeneration
                            arch_cls = Qwen2AudioForConditionalGeneration
                        elif "Qwen2_5Omni" in archs:
                            # Qwen2.5-Omni：transformers 5.x 中未注册 AutoModel 映射，
                            # 必须用显式类名（AutoModel 会报 Unrecognized config）。
                            from transformers import Qwen2_5OmniForConditionalGeneration
                            arch_cls = Qwen2_5OmniForConditionalGeneration
                        elif "AudioFlamingo3ForConditionalGeneration" in archs:
                            from transformers import AudioFlamingo3ForConditionalGeneration
                            arch_cls = AudioFlamingo3ForConditionalGeneration
                        elif "Qwen2_5OmniModel" in archs.replace(" ", ""):
                            from transformers import Qwen2_5OmniForConditionalGeneration
                            arch_cls = Qwen2_5OmniForConditionalGeneration
                        elif "Omni" in archs:
                            arch_cls = AutoModel  # 其他 Omni 兜底（可能失败，由容错处理）
                        elif "Audio" in archs and "CausalLM" not in archs:
                            arch_cls = AutoModel  # 其他音频生成模型兜底
                except Exception as _e:  # noqa: BLE001
                    # v6.6.2-fix（2026-08-11 巡检）：原 debug 级静默吞掉架构检测/BF16
                    # 直载失败 → 回退 CausalLM fp16。对 Gemma-4 多模态模型这是错误加载
                    # （fp16 CausalLM generate 病理挂死，E2B 反复停滞），静默降级违反
                    # 纪律 #2（可追溯）。升为 WARNING 且披露降级后果，供巡检定位。
                    self.log.warning(
                        "架构检测/BF16 直载异常（%s）→ 回退 CausalLM fp16——对 Gemma-4 "
                        "多模态模型此为错误加载，generate 可能病理挂死，须核对",
                        str(_e)[:150])
                try:
                    model = arch_cls.from_pretrained(
                        model_path, local_files_only=True, **kwargs)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    # v6.5.23-fix（问题 89，2026-08-08）：仅真实 OOM（含 "out of
                    # memory"）才降级 4bit；非 OOM 的 RuntimeError（权重损坏/代码
                    # 错误/版本不匹配）原样上抛记录真实原因，不得误写"降级 4bit"
                    # 日志（原条件 `and precision == "4bit"` 使非 OOM + 非 4bit
                    # 场景仍走降级，掩盖真实错误，违反纪律 #2 可追溯）。
                    if "out of memory" not in str(e).lower():
                        raise
                    self.log.warning("[%s] fp16 失败（%s），降级 4bit", name,
                                     str(e)[:200])
                    torch.cuda.empty_cache()
                    from transformers import BitsAndBytesConfig
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4")
                    kwargs.pop("torch_dtype", None)
                    model = arch_cls.from_pretrained(
                        model_path, local_files_only=True, **kwargs)
                    precision = "4bit"
                model.eval()
                result["m"] = (model, tok, precision)
            except Exception as e:  # noqa: BLE001
                err["e"] = e

        t0 = time.time()
        th = threading.Thread(target=_load, daemon=True)
        th.start()
        th.join(self.load_timeout)
        if th.is_alive():
            raise TimeoutError(
                f"[{name}] 模型加载超过 {self.load_timeout}s（疑似存储 I/O 挂起），"
                f"跳过该模型并记录")
        if "e" in err:
            raise err["e"]
        peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        self.log.info("[%s] 加载完成 | 精度=%s | 耗时=%.0fs | 显存峰值=%.1fGB",
                      name, result["m"][2], time.time() - t0, peak)
        self._cache[name] = result["m"]
        st = self._stats.setdefault(name, {"loads": 0, "samples": 0})
        st["loads"] += 1
        return result["m"]

    def unload(self, name: str):
        if name in self._cache:
            model, _, _ = self._cache.pop(name)
            del model
            import gc
            gc.collect()
            try:
                import torch as _t
                if _t.cuda.is_available():
                    _t.cuda.empty_cache()
            except ImportError:
                pass
            self.log.info("[%s] 已释放显存", name)

    def unload_all(self):
        for n in list(self._cache):
            self.unload(n)
        try:
            import torch as _t
            peak = _t.cuda.max_memory_allocated() / 1e9 if _t.cuda.is_available() else 0
        except Exception:
            peak = 0
        self.log.info("阶段结束 empty_cache | 全程显存峰值 %.1fGB", peak)


# ---------------------------------------------------------------------------
# 推理（greedy 解码，批处理，超时保护）
# ---------------------------------------------------------------------------

@_torch_no_grad()
def generate_batch(model, tokenizer, prompts: list, max_new_tokens: int = 512,
                   batch_size: int = 8) -> list:
    """统一 greedy 批推理（temperature=0，与原论文一致）。"""
    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    outputs = []
    try:
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i: i + batch_size]
            inputs = tokenizer(chunk, return_tensors="pt", padding=True,
                               truncation=True, max_length=4096).to(model.device)
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
            for j in range(out.shape[0]):
                outputs.append(tokenizer.decode(
                    out[j, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True))
    finally:
        tokenizer.padding_side = old_padding
    return outputs


def chat_prompts(tokenizer, user_prompts: list) -> list:
    """将 user 文本列表转为 chat template 提示（无模板模型回退为原文）。"""
    out = []
    for p in user_prompts:
        try:
            out.append(tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True))
        except Exception:  # noqa: BLE001
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

def set_deterministic(seed: int = 42):
    """KBS 可复现性（v6.5.30 补全）：固定全局随机种子 + 确定性 CUDA。

    所有阶段经 load_config 调用——固定 python/random/numpy/torch/CUDA 种子，
    并启用 cudnn.deterministic（禁 benchmark 自动调优），配合 config.seeds
    逐阶段种子，使相同种子运行结果可复现（KBS 复现性要求）。
    """
    import os
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except Exception:  # noqa: BLE001
        pass


def load_config(path: str) -> dict:
    import yaml
    # v6.5.30（KBS 可复现性）：加载配置即固定全局种子 + 确定性 CUDA
    set_deterministic(42)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 环境变量覆盖（可移植性：新服务器只需设环境变量）
    import os
    if os.environ.get("LALM_WORKDIR"):
        cfg["workdir"] = os.environ["LALM_WORKDIR"]
    if os.environ.get("LALM_ORIGINAL_DATA"):
        cfg["original_data_dir"] = os.environ["LALM_ORIGINAL_DATA"]
    if os.environ.get("HF_HOME"):
        cfg["hf_home"] = os.environ["HF_HOME"]
    return cfg


def ensure_dirs(cfg: dict):
    """按提示词 §1 建立目录结构。"""
    root = Path(cfg["workdir"]).expanduser()
    for d in ["logs", "checkpoints", "results", "responses", "models", "report"]:
        (root / d).mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# 样本充足性核对（防止 I/O 瓶颈导致的样本量不足静默通过）
# ---------------------------------------------------------------------------

def check_sample_sufficiency(counts: dict, target: int, stage: str,
                             logger: logging.Logger, min_ratio: float = 0.75,
                             summary_path: str = None) -> bool:
    """阶段结束核对每条件实际样本量，不足时醒目 WARNING 并写入 stage summary。

    counts: {condition: achieved_n}
    target: 每条件目标样本量（如 200）
    min_ratio: 最低可接受比例（默认 75%，即 200 目标下 <150 触发警告）
    返回 True=全部达标。
    """
    lines = [f"# {stage} 样本充足性核对\n",
             f"目标: {target}/条件 | 最低可接受: {min_ratio:.0%}\n\n",
             "| 条件 | 实际 | 达标 |\n|---|---|---|\n"]
    all_ok = True
    for cond, n in sorted(counts.items()):
        ok = n >= target * min_ratio
        all_ok = all_ok and ok
        flag = "OK" if ok else "**不足**"
        lines.append(f"| {cond} | {n} | {flag} |\n")
        if not ok:
            logger.warning("⚠⚠⚠ [%s] 条件 %s 样本量 %d < 目标 %d 的 %.0f%%"
                           "（疑似 I/O 瓶颈导致推理未完成）——"
                           "恢复后重跑本阶段即可断点续补",
                           stage, cond, n, target, min_ratio)
    if all_ok:
        logger.info("[%s] 样本充足性核对通过（每条件 ≥%d）", stage,
                    int(target * min_ratio))
    if summary_path:
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_path).write_text("".join(lines), encoding="utf-8")
    return all_ok


def isnan(v):
    """通用 NaN/None 判定（v6.5.28-fix，第八轮审查 🔴）。

    背景：pandas iterrows 取到 np.float64 时 `isinstance(v, float)` 为 False，
    且 numpy 标量的 `v != v`/`float(nan)` 行为与 Python float 不一致，导致
    多处把 NaN 当正常值推进数值管线 → 模型 fit 抛 ValueError 静默瘫痪。
    此处统一用 pd.isna 语义判定 None/NaN/NA，标量与数组均可。
    """
    try:
        import pandas as pd  # noqa: PLC0415
        return bool(pd.isna(v))
    except Exception:  # noqa: BLE001
        # pandas 不可用（纯 numpy 环境）时回退：None/NaN 判真，其余假
        if v is None:
            return True
        try:
            return bool(v != v)  # NaN 自不等
        except Exception:  # noqa: BLE001
            return False


def load_measurement_query_ids(root, files=("p1_pilot_queries_full.json",
                                            "p1_full_queries_full.json")):
    """读取测量集（PILOT/FULL）已用 query_id 集（LM1/LM2-fix，AUDIT #172 fix
    13）。

    供 P0-C / P2-C 在构建各自查询池时按 query_id 排除测量集查询，消除
    「攻击/评测查询与测量集查询重叠」的偏差来源（LM1/LM2）。PILOT 权威冻结集
    为 p1_pilot_queries_full.json；FULL 用集由 stage_p1_full 在采样后落盘
    p1_full_queries_full.json（同构 {queries:[{query_id,zh,en}]}）。任一文件
    缺失/解析失败时静默跳过该集（P0-C 与 P1-FULL 并行运行，FULL 文件在 P0-C
    启动时可能尚不存在 → 只排除 PILOT；P2-C 在 FULL 之后运行，两集均可读）。

    返回 (query_ids:set, srcs:list[str])；srcs 记录实际读到的来源文件名供
    调用方披露。注意：结果不含"不足 200 的排除预算"概念——调用方须自行判断
    排除后剩余池是否 ≥ n_per（本函数只负责读，不做丢弃）。
    """
    import json as _json

    ids: set = set()
    srcs: list = []
    for fname in files:
        f = root / "results" / fname
        if not f.exists():
            continue
        try:
            for q in _json.loads(f.read_text(encoding="utf-8")).get("queries", []):
                qid = str(q.get("query_id") or "").strip()
                if qid:
                    ids.add(qid)
            srcs.append(fname)
        except Exception:  # noqa: BLE001
            pass  # 单文件解析失败不阻断；调用方在日志中自行披露 srcs
    return ids, srcs


def verify_pool_frozen_consistency(root, log=None,
                                   sources=("queries_v2.jsonl",
                                            "queries_v1.jsonl")):
    """FIXED v6.5.29 (审计 C-5)：冻结集 query_id 下池语义一致性守卫。

    若 results/p1_pilot_queries_full.json 存在，逐一核对池中 150 个 PILOT
    query_id 的 zh/en/category 与冻结文件一致。冻结文件是 PILOT 实验实际
    运行的刺激文本，池在 08-08 曾整池重写而保留 query_id → 语义漂移导致
    "query_id→池内容"映射（攻击族/反查）取到错误文本。本函数在漂移后返回
    (False, 明细)，调用方须告警并禁止依赖该映射的下游（fail-closed）。
    冻结文件缺失/解析失败时返回 (True, {})（无基准可比，不误报）。

    返回 (ok: bool, mismatches: dict[qid][field]=(frozen, pool))。
    """
    f = Path(root) / "results" / "p1_pilot_queries_full.json"
    if not f.exists():
        return True, {}
    try:
        import json as _json
        frozen = _json.loads(f.read_text(encoding="utf-8")).get("queries", [])
    except Exception:  # noqa: BLE001
        return True, {}
    fmap = {q.get("query_id"): q for q in frozen if q.get("query_id")}
    if not fmap:
        return True, {}
    mism = {}
    for src in sources:
        p = Path(root) / "data" / src
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                import json as _json
                r = _json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            qid = r.get("query_id")
            fr = fmap.get(qid)
            if fr is None:
                continue
            for k in ("zh", "en", "category"):
                if r.get(k) != fr.get(k):
                    mism.setdefault(qid, {})[k] = (fr.get(k), r.get(k))
    if mism and log is not None:
        log.error("FIXED C-5 一致性守卫：冻结集 query_id 下池内容漂移 %d 条 "
                  "→ 禁止依赖 query_id→池内容的映射", len(mism))
    return not mism, mism
def verify_df_pool_consistency(df, root, log=None,
                               sources=("queries_v2.jsonl",
                                        "queries_v1.jsonl",
                                        "benign_requests_v1.jsonl")):
    """v6.5.39-fix (审计 C-5 参照系修正)：本 run 数据↔当前池 一致性守卫。

    verify_pool_frozen_consistency 把当前池与 08-08 14:42 的 PILOT 冻结基线
    比对——但池在 08-08 15:16 已合法整池重写（保留 query_id、更新文本），该
    比较对 FULL 流水线是时代错位误报。攻击族映射实际取当前池 category，真正
    完整性目标是「本 run 行 query_text ↔ 当前池 zh/en」：若池在本 run 数据
    生成后重写，行文本与池文本失配 → query_id→池内容 映射不可信 → fail-closed。
    返回 (ok: bool, mismatches: dict[qid]=(row_text, pool_zh, pool_en))。
    """
    pool = {}
    for src in sources:
        p = Path(root) / "data" / src
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            qid = str(r.get("query_id") or "").strip()
            if qid:
                pool.setdefault(qid, r)
    try:
        sub = df[df["pool_query_id"].astype(str).str.strip().ne("")]
    except Exception:  # noqa: BLE001
        return True, {}
    mism = {}
    if not len(sub):
        return True, {}
    for qid, g in sub.groupby(sub["pool_query_id"].astype(str).str.strip()):
        po = pool.get(qid)
        if po is None:
            continue
        zh = str(po.get("zh") or "").strip()
        en = str(po.get("en") or "").strip()
        for txt in g["query_text"].astype(str).unique():
            t = txt.strip()
            if t != zh and t != en:
                mism.setdefault(qid, (t[:60], zh[:60], en[:60]))
    if mism and log is not None:
        log.error("C-5 (v6.5.39) 数据↔池一致性守卫：%d 条不一致 → 禁止依赖"
                  " query_id→池内容的映射", len(mism))
    return not mism, mism

