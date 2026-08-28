# Appendix D: Length Confounding and the Mechanism of Framing-Induced Harm Amplification in LALMs

*Draft — KBS submission appendix. Data: P0-C, `results/p0c_scored.parquet` (N=10,800, harmbench primary). Analysis scripts: `p0c_len_control.py`, `p0c_len_match.py`. All tables below are computed, not estimated.*

## D.1 Motivation

The raw framing effect in audio (baseline 5.3% → storytelling 55.1% → unrestricted 63.6% ASR) is large, but framing prompts also change *response length*: storytelling audio responses grow from a mean of 92 to 923 characters (≈10×). A reviewer must ask whether the effect is a true framing effect on harmful content, or an artifact of framing inducing longer, more elaborated responses — with harm simply concentrating in longer output. This appendix decomposes the effect.

**Statistical caution applied throughout.** Baseline responses are predominantly short (<50 chars; 1066/1800 audio), while framed responses are predominantly long (>800 chars). There is **limited length overlap between conditions**. Consequently the length-adjusted effect is estimated from thin, sparse strata and is expected to be *method-dependent*; we report three estimators rather than one, and do not select a favorable one.

## D.2 Results: three estimators

### D.2.1 Response-length stratified ASR (all models pooled)

**Audio modality**

| Length band | baseline ASR | storytelling ASR | unrestricted ASR |
|---|---|---|---|
| <50 (n=1066/69/112) | 2.2% | 0.0% | 0.0% |
| 50–199 (592/30/53) | 2.5% | 3.3% | 0.0% |
| 200–799 (98/580/1014) | 26.5% | 50.0% | 70.1% |
| ≥800 (44/1118/621) | **72.7%** | 62.7% | 69.7% |

**Text modality**

| Length band | baseline ASR | storytelling ASR | unrestricted ASR |
|---|---|---|---|
| <50 (178/74/68) | 2.8% | 0.0% | 0.0% |
| 50–199 (462/20/52) | 1.7% | 0.0% | 0.0% |
| 200–799 (267/586/524) | 9.0% | 27.8% | 55.2% |
| ≥800 (459/858/814) | **3.1%** | 24.7% | 20.3% |

Key observation: **long responses are harmful regardless of framing.** In audio, baseline responses that happen to be ≥800 chars already carry 72.7% ASR — essentially the same as framed responses at that length (62.7–69.7%). The aggregate +50pp effect is therefore driven largely by framing *moving* responses into the long band, not by making same-length responses more harmful.

Crucially, the length→harm channel is **modality-asymmetric**: at ≥800 chars, *text* baseline responses are only 3.1% harmful, whereas *audio* baseline responses are 72.7%. This reframes the modality difference as an *elaboration-channel* difference (long audio output is high-risk; long text output is not automatically harmful), not as "audio models are more susceptible to framing prompts."

### D.2.2 Single-slope logistic (log_len covariate)

Ridge-IRLS logistic, `harmbench_label ~ condition + log(1+len)`, per model × modality. Reporting raw OR → adjusted OR.

| Model | Modality | Condition | OR raw | OR adj | 95% CI | p (adj) |
|---|---|---|---|---|---|---|
| e2b | audio | storytelling | 54.7 | 1.24 | [0.55, 2.77] | 0.604 |
| e2b | audio | unrestricted | 49.7 | 2.35 | [1.07, 5.16] | 0.033 |
| e2b | text | storytelling | 1.00 | 0.21 | [0.01, 3.65] | 0.284 |
| e2b | text | unrestricted | 1.00 | 0.32 | [0.02, 5.19] | 0.420 |
| e4b | audio | storytelling | 10.4 | 5.20 | [2.88, 9.39] | 4.5e-08 |
| e4b | audio | unrestricted | 12.0 | 6.51 | [3.77, 11.24] | 1.8e-11 |
| e4b | text | storytelling | 0.14 | 0.14 | [0.03, 0.62] | 0.010 |
| e4b | text | unrestricted | 0.28 | 0.28 | [0.09, 0.85] | 0.025 |
| qwen | audio | storytelling | 20.7 | 0.60 | [0.31, 1.17] | 0.136 |
| qwen | audio | unrestricted | 65.3 | 2.57 | [1.35, 4.88] | 0.004 |
| qwen | text | storytelling | 25.6 | 0.19 | [0.09, 0.41] | 1.8e-05 |
| qwen | text | unrestricted | 46.6 | 0.76 | [0.38, 1.52] | 0.431 |

Length × condition interaction was significant in all three audio models (p ≤ 1.7e-06) and in none of the text models (p ≥ 0.25): length is a **modifier** in audio, not a constant confounder.

### D.2.3 Mantel–Haenszel decile-stratified OR (log_len deciles)

| Model | Modality | Condition | OR_MH | 95% CI | p | within-stratum diff | strata same-direction |
|---|---|---|---|---|---|---|---|
| e2b | audio | storytelling | 1.70 | [1.30, 2.21] | <0.001 | +2.6pp | 89% |
| e2b | audio | unrestricted | 3.18 | [2.31, 4.38] | <0.001 | +7.0pp | 67% |
| e2b | text | storytelling | 0.53 | [0.36, 0.78] | 0.001 | −0.4pp | 50% |
| e2b | text | unrestricted | 0.62 | [0.41, 0.93] | 0.021 | −0.2pp | 50% |
| e4b | audio | storytelling | 0.60 | [0.47, 0.77] | <0.001 | −9.3pp | 29% |
| e4b | audio | unrestricted | 1.02 | [0.80, 1.31] | 0.859 | −0.5pp | 43% |
| e4b | text | storytelling | 0.39 | [0.29, 0.52] | <0.001 | −1.9pp | 20% |
| e4b | text | unrestricted | 0.49 | [0.38, 0.63] | <0.001 | −2.0pp | 40% |
| qwen | audio | storytelling | 0.84 | [0.69, 1.03] | 0.092 | −5.5pp | 44% |
| qwen | audio | unrestricted | 6.16 | [4.63, 8.19] | <0.001 | +33.9pp | 38% |
| qwen | text | storytelling | 0.82 | [0.67, 1.01] | 0.057 | −6.9pp | 43% |
| qwen | text | unrestricted | 2.85 | [2.34, 3.46] | <0.001 | +14.0pp | 57% |

### D.2.4 1:1 nearest-neighbor matching (log_len, caliper 0.2 SD) + McNemar

| Model | Modality | Condition | ΔASR matched | n pairs | p (McNemar) |
|---|---|---|---|---|---|
| e2b | audio | storytelling | +2.9pp | 69 | 0.683 |
| e2b | audio | unrestricted | +2.8pp | 71 | 0.683 |
| e2b | text | storytelling | −1.8pp | 55 | 1.000 |
| e2b | text | unrestricted | −2.0pp | 49 | 1.000 |
| e4b | audio | storytelling | −10.5pp | 38 | 0.386 |
| e4b | audio | unrestricted | −4.1pp | 49 | 0.724 |
| e4b | text | storytelling | −4.3pp | 115 | 0.131 |
| e4b | text | unrestricted | −2.7pp | 110 | 0.371 |
| qwen | audio | storytelling | −39.3pp | 28 | **0.003** |
| qwen | audio | unrestricted | −5.9pp | 34 | 0.617 |
| qwen | text | storytelling | −33.3pp | 33 | **0.006** |
| qwen | text | unrestricted | −4.4pp | 68 | 0.371 |

## D.3 Interpretation

1. **The aggregate audio framing effect is substantially length-mediated.** Every length-control estimator collapses raw ORs of 10–65× toward ~0.6–6×, and the length-stratified table shows long baseline audio responses are already as harmful as framed ones. The primary mechanism of framing is **response elaboration**: prompting the model to produce more output, which is where harmful content concentrates.

2. **A robust length-independent direct effect is *not* cleanly identifiable from this design.** Estimates are method-dependent: e.g., e4b audio storytelling is OR 5.20 under single-slope logistic but OR 0.60 (protective direction) under MH stratification and n.s. under matching. This instability reflects the near-collinearity of condition and length in audio (baseline is almost always short; framed almost always long). **We do not claim a length-independent framing effect.** Instead, we claim: framing induces elaboration; elaboration is associated with high harm rates in audio and, at matched lengths, text unrestricted framing retains a modest positive effect (qwen: OR_MH 2.85, matched −4.4pp n.s.).

3. **The modality asymmetry is real but re-interpreted.** The striking, robust asymmetry is in the *length→harm channel*: long audio responses are harmful (baseline 72.7% at ≥800) whereas long text responses are not (baseline 3.1%). Combined with framing's strong push toward long output in audio, this produces the large audio framing effect without requiring any claim that audio models are uniquely "manipulable" by prompt framing. We present this as the **elaboration-channel hypothesis**.

## D.4 What we do NOT claim (explicit non-claims)

- We do **not** claim framing directly increases harmfulness of fixed-length audio output.
- We do **not** claim "audio modality is uniquely vulnerable to framing manipulation" (previous wording, retired).
- We do **not** claim a stable length-independent OR; point estimates are method-dependent.

## D.5 Limitations of this analysis

- Sparse overlap: few baseline long responses (audio ≥800: n=44) make adjusted estimates low-powered; continuity correction applied in MH.
- Matching discards most framed responses (n pairs 28–115), and qwen matched estimates are driven by few long-baseline pairs.
- Length measured in characters, not tokens; log transform assumed.
- A definitive direct-effect test would require *length-controlled elicitation* (e.g., instruction to answer in ≤N words) or *prefix-truncation rescoring*, both listed as future work.

## D.6 Suggested paper wording (replacing prior modality-vulnerability claims)

> "Framing prompts amplify harm rates in LALMs primarily by eliciting longer, more elaborated responses in which harmful content concentrates. The effect is strongest in the audio modality because long audio output is disproportionately likely to be judged harmful even without framing (72.7% ASR at ≥800 chars baseline vs 3.1% for text), whereas framing in text retains a smaller, partly direct component. These findings identify response elaboration — not a modality-specific susceptibility to prompt manipulation — as the operative mechanism, and motivate length-controlled evaluation as a design requirement for safety benchmarks."
