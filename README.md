# Cross-Modal Safety Inconsistency in Large Audio-Language Models

**Measurement, Mechanism, and Knowledge-Driven Multi-Source Fusion**

[![code: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![data/paper: CC BY 4.0](https://img.shields.io/badge/data%2Fpaper-CC_BY_4.0-lightgrey.svg)](LICENSE-CC-BY-4.0)
[![status: under review](https://img.shields.io/badge/status-under_review-orange.svg)](#citation)

Official code, data, and results for the manuscript submitted to
*Knowledge-Based Systems* (Elsevier). This repository contains the full experiment
pipeline, benchmark data, analysis code, frozen results, and LaTeX source required to
understand and reproduce the reported findings.

## Paper

- **Title:** Cross-Modal Safety Inconsistency in Large Audio-Language Models: Measurement, Mechanism, and Knowledge-Driven Multi-Source Fusion
- **Manuscript (compiled PDF):** [`paper/main.pdf`](paper/main.pdf)
- **LaTeX source:** [`paper/main.tex`](paper/main.tex) + [`paper/sections/`](paper/sections)
- **Status:** under review at *Knowledge-Based Systems*

## Overview

Large audio-language models (LALMs) are deployed in safety-sensitive contexts, yet the
safety implications of *adversarial framing* — repackaging a fixed request through
narrative structure, role-play, or acoustic style — remain poorly understood. We treat
this as a knowledge-system problem and propose a knowledge-driven methodology in which a
safe decision fuses heterogeneous knowledge sources: request intent, narrative structure,
acoustic style, and scoring uncertainty.

## Key results

| Finding | Value | Artifact |
|---------|-------|----------|
| Narrative structure raises attack success | **27.9 pp** (95% CI [25.0, 30.9]; OR 16.65) | [`results/p1_full_effects.json`](results/p1_full_effects.json) |
| MSRF fusion ROC-AUC (deployment frame) | **0.9784** (0.9455 ± 0.030 CV) | [`results/msrf_evaluation.json`](results/msrf_evaluation.json) |
| MSRF parameter count | **225** | [`report/msrf_evaluation.md`](report/msrf_evaluation.md) |
| Benign false-positive rate (calibrated threshold) | **75.9%** | [`results/benign_control/benign_fpr_analysis.json`](results/benign_control/benign_fpr_analysis.json) |
| Adaptive-attack decay | reported honestly | [`results/p2c4_defense_decay.json`](results/p2c4_defense_decay.json) |
| Scorer validation on public benchmarks | judge_big acc 0.8555, etc. | [`results/scorer_accuracy.json`](results/scorer_accuracy.json) |

Every reported number is traceable to a per-experiment report under
[`report/`](report) and its corresponding frozen result under [`results/`](results).

## Repository layout

| Path | Contents |
|------|----------|
| `paper/` | LaTeX source (`main.tex`, `sections/`), compiled `main.pdf`, `references.bib`, manuscript figures |
| `results/` | Frozen per-experiment result JSONs backing every number in the paper |
| `report/` | Per-experiment human-readable reports and summary CSVs/JSON |
| `data/` | Attack queries, recipes, templates, benign requests, and public benchmark samples |
| `gold/` | Human-annotation protocol, strata plan, and annotation set |
| `figures/` | High-resolution figures (main + supplementary), PDF and PNG |
| `stage_*.py` | Pipeline stages: `L` (novelty) → `D` (data) → `P0` (scoring) → `P1` (pilot/full) → `P2` (MSRF / baselines / adaptive) → `P3` / `F` / `R` |
| `gpu1_s*.py` | Individual GPU experiments (convergence, determinism, cross-family, modality, power, FDR, etc.) |
| `s_*.py`, `s40*.py`, `s41*.py` | Scoring, aggregation, and benchmark-table generation |
| `common_utils.py`, `scorer_utils.py`, `model_cache.py`, `agreement_utils.py` | Shared libraries and model registry |
| `gate_g1.py`, `gate_g2.py` | Pre-registered decision gates |
| `pipeline.sh`, `pipeline_config.yaml` | Orchestration script and global configuration (model registry, seeds, endpoints) |
| `RESEARCH_PROTOCOL.md`, `STAGE_CONTRACTS.md` | Research protocol (authoritative) and stage contracts (DAG, I/O, gate format) |
| `requirements_frozen.txt` | Frozen Python dependencies |

## Installation

```bash
git clone https://github.com/chenyongjingg/lalm-framing-kbs.git
cd lalm-framing-kbs
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements_frozen.txt
```

Model weights and API endpoints are configured in `pipeline_config.yaml`; the LALM and
judge/guard models referenced by the experiments are loaded from their public
repositories (e.g., Hugging Face) or configured endpoints.

## Reproducing the results

The pipeline is orchestrated by `pipeline.sh` and configured in `pipeline_config.yaml`.
Stages are chained as:

```
L → D → P0 → P1-PILOT → [G1] → P1-FULL ∥ P0-C → P2 (MSRF) → P2-C → [G2] → P2-B → F → R
```

Because the full experiment suite runs several LALMs and judge/guard models on GPUs, the
complete raw outputs (response sets, model checkpoints) are large and are **not**
included here; they are available from the authors on request. The included code, data,
frozen results, and per-experiment reports are sufficient to reconstruct each reported
number and to re-run any single stage on your own hardware.

## Data availability

All pipeline artifacts, configurations, frozen seeds, and result files are released in
this public repository; no pre-trained guard weights are redistributed. The intent-classifier
training data is synthetic.

## Notes for reviewers

- `paper/main.pdf` is the compiled manuscript; `paper/main.tex` + `paper/sections/` are
  self-contained LaTeX sources.
- The five result figures referenced by the manuscript live in `paper/figures/`
  (`adaptive_decay`, `benign_fpr_boundary`, `factorial_forest`, `msrf_roc_pr`,
  `pcsd_heatmap`); architecture diagrams are drawn natively in TikZ.
- All reported numbers follow the frozen values in `RESEARCH_PROTOCOL.md`, the
  per-experiment reports under `report/`, and the JSONs under `results/`.

## Citation

The manuscript is under review. If you use this work, please cite:

```bibtex
@article{chen2026crossmodal,
  title   = {Cross-Modal Safety Inconsistency in Large Audio-Language Models:
             Measurement, Mechanism, and Knowledge-Driven Multi-Source Fusion},
  author  = {Chen, Yongjin and Zhu, Songze},
  journal = {Knowledge-Based Systems},
  year    = {2026},
  note    = {Under review}
}
```

## License

This repository is dual-licensed:

- **Code** (`*.py`, `*.sh`) — [MIT License](LICENSE)
- **Paper text, figures, and data** — [CC BY 4.0](LICENSE-CC-BY-4.0)
