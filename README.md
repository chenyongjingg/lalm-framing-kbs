# Cross-Modal Safety Inconsistency in Large Audio-Language Models

**Measurement, Mechanism, and Knowledge-Driven Multi-Source Fusion**

This repository accompanies the manuscript submitted to *Knowledge-Based Systems*. It
contains the full experiment pipeline, benchmark data, analysis code, and paper source
needed to understand and reproduce the reported results.

## Overview

Large audio-language models (LALMs) are deployed in safety-sensitive contexts, yet the
safety implications of *adversarial framing* — repackaging a fixed request through
narrative structure, role-play, or acoustic style — remain poorly understood. We treat
this as a knowledge-system problem and propose a knowledge-driven methodology in which a
safe decision fuses heterogeneous knowledge sources: request intent, narrative structure,
acoustic style, and scoring uncertainty.

Key findings:

- In a controlled paired factorial experiment ($E_t \times N \times R \times A_s$,
  24 cells; 16,200 response cells; three LALMs; two languages), narrative structure
  raises attack success by **27.9 pp** (95% CI [25.0, 30.9]; odds ratio 16.65
  [14.10, 19.67]).
- Paired cross-modal safety divergence (PCSD) is **model-dependent**, so audio
  vulnerability is not intrinsic to the modality.
- **MSRF** — a knowledge-driven multi-source fusion framework — fuses the four signal
  families through calibrated out-of-fold scores in an auditable **225-parameter** layer,
  reaching **0.9784 ROC-AUC** on the deployment frame (0.9455 ± 0.030 cross-validation).
- We report limitations honestly, including a 75.9% benign false-positive rate under
  independent benign control and adaptive-attack decay of the fixed-threshold detector.

## Repository layout

| Path | Contents |
|------|----------|
| `paper/` | LaTeX source (`main.tex`, `sections/`), compiled `main.pdf`, `references.bib`, and figures |
| `stage_*.py` | Pipeline stages: `L` (novelty) → `D` (data) → `P0` (scoring) → `P1` (pilot/full) → `P2` (MSRF / baselines / adaptive) → `P3` / `F` / `R` |
| `gpu1_s*.py` | Individual GPU experiments (convergence, determinism, cross-family, modality, power, FDR, etc.) |
| `s_*.py`, `s40*.py`, `s41*.py` | Scoring, aggregation, and benchmark-table generation |
| `scorer_utils.py`, `common_utils.py`, `model_cache.py`, `agreement_utils.py`, `gate_g1.py`, `gate_g2.py` | Shared libraries, model registry, and stage gates |
| `pipeline.sh`, `pipeline_config.yaml` | Orchestration script and global configuration (model registry, seeds, endpoints) |
| `data/` | Attack queries, recipes, templates, benign requests, and benchmark samples |
| `gold/` | Human-annotation protocol, strata plan, and annotation set |
| `report/` | Per-experiment result reports and summary CSVs/JSON |
| `RESEARCH_PROTOCOL.md`, `STAGE_CONTRACTS.md` | Research protocol (authoritative) and stage contracts (DAG, I/O, gate format) |
| `requirements_frozen.txt` | Frozen Python dependencies |

## Reproducing the results

The pipeline is orchestrated by `pipeline.sh` and configured in `pipeline_config.yaml`
(model registry, seeds, endpoints). Stages are chained as:

```
L → D → P0 → P1-PILOT → [G1] → P1-FULL ∥ P0-C → P2 (MSRF) → P2-C → [G2] → P2-B → F → R
```

Because the full experiment suite runs several LALMs and judge/guard models on GPUs, the
complete raw outputs (response sets, model checkpoints) are large and are **not** included
here; they are available from the authors on request. The included code, data, and
per-experiment summary reports are sufficient to reconstruct each reported number and to
re-run any single stage on your own hardware.

## Notes for reviewers

- `paper/main.pdf` is the compiled manuscript; `paper/main.tex` + `paper/sections/` are
  self-contained LaTeX sources.
- The five result figures referenced by the manuscript live in `paper/figures/`
  (`adaptive_decay`, `benign_fpr_boundary`, `factorial_forest`, `msrf_roc_pr`,
  `pcsd_heatmap`); the two architecture diagrams are drawn natively in TikZ and their
  rendered PDFs/PNGs are also provided.
- All reported numbers follow the frozen values in `RESEARCH_PROTOCOL.md` and the
  per-experiment reports under `report/`.

## License

This repository is dual-licensed:

- **Code** (`*.py`, `*.sh`) — [MIT License](LICENSE)
- **Paper text, figures, and data** — [CC BY 4.0](LICENSE-CC-BY-4.0)

## Citation

If you use this work, please cite the corresponding manuscript (details to be finalized
upon acceptance).
