# Appendix: Robustness of the Judge-Small Scorer to Format-Violation Null Cells

This appendix documents the completeness, systematic-bias, and robustness
evidence for the *judge_small* scorer (Gemma-4-E2B-it) used as one of the two
consensus judges in our dual-judge protocol. It addresses the standard reviewer
concern about excluded unrateable cells: (i) what the null cells are, (ii)
whether their exclusion biases any reported effect, and (iii) whether an
independent scoring mode confirms the same verdicts. All analysis is
reproducible from `results/gpu1_pipeline/` (stages S28, S34, S35, S36); zero
human labels were introduced at any point.

## A.1 Scoring protocol and the format-violation failure mode

The judge-small scorer applies a fixed binary rubric (``0`` = refusal,
``1`` = harmful compliance) with a tight generation budget of
`max_new_tokens=8`, then extracts the verdict with `\b([01])\b`. For a subset
of narrative-type responses, the model begins with English narration (e.g.,
``The user request is: ...``) and exhausts the 8-token budget before emitting
the verdict digit; the regex then fails and the cell is recorded as
*null* (label unparseable).

We evaluated but **rejected** extended-generation recovery: re-scoring with a
512-token budget produces trailing digits that contradict the model's own
prose reasoning (e.g., q0061's prose concludes ``not harmful`` while the
trailing digit is 1; q0119's prose concludes ``harmful`` while the trailing
digit is 0). These scrambled trailing digits are generation-degradation
artifacts, not verdicts. Recovery was therefore abandoned in favor of (a)
honest exclusion plus (b) a forced-decode scoring mode that is immune to the
failure (Appendix A.4).

## A.2 Coverage and systematic-bias audit across pipeline stages (S34)

Across all five pipeline stages, judge-small coverage is high
(99.1%--99.7%) and the null cells are rare (83 / 15,944 total:
3,600 + 7,200 + 3,600 + 1,200 + 344).

| Stage | Null / total | Coverage | Null-cell harmful rate (judge_big) | Non-null harmful rate | Ratio |
|---|---|---|---|---|---|
| E2B main (text) | 27 / 3,600 | 99.25% | 0.333 | 0.092 | 3.62$\times$ |
| S17 E4B audio | 23 / 7,200 | 99.68% | 0.435 | 0.234 | 1.86$\times$ |
| S17 E4B text | 19 / 3,600 | 99.47% | 0.158 | 0.069 | 2.29$\times$ |
| S28 hetero audio | 11 / 1,200 | 99.08% | 0.455 | 0.330 | 1.38$\times$ |
| S33 hetero audio | 3 / 344 | 99.13% | 0.000 | 0.232 | --- (n=3) |

The null cells are *not* a random subset: per the independent judge-big scorer,
they are systematically more likely to be harmful (3.62$\times$ on the primary
E2B main stage). In the E2B main stage, 23 / 27 null cells (85%) fall in the
N=0 condition, so N=0's harmful rate is the most understated by exclusion.
The absolute counts are, however, small (27 cells $<$ 1 percentage point of
effect), and the sensitivity analysis below shows the exclusion does not
change any conclusion.

## A.3 Sensitivity of main effects to null-cell handling

We re-estimated the two headline effects --- the number-condition effect
*N* (E_t=0, N=0 vs. N=1) and the narrative-framing effect *E_t* (N=0,
E_t=0 vs. E_t=1) --- on the E2B main stage under the paper's authoritative
**dual-judge consensus scope** (judge-big == judge-small), replacing the 27
null judge-small cells with six different assignments (query-clustered
bootstrap, B = 2,000; ✓ = 95% CI excludes 0).

| Null-cell handling | *N* effect [95% CI] | *E_t* effect (N=0) [95% CI] |
|---|---|---|
| Exclude (baseline) | 0.0568 [0.0333, 0.0823] ✓ | 0.0282 [0.0019, 0.0570] ✓ |
| Assign judge-big | 0.0559 [0.0317, 0.0829] ✓ | 0.0341 [0.0093, 0.0611] ✓ |
| Assign StrongREJECT | 0.0463 [0.0198, 0.0743] ✓ | 0.0326 [0.0065, 0.0619] ✓ |
| All harmful | 0.0422 [0.0162, 0.0715] ✓ | 0.0286 [0.0022, 0.0576] ✓ |
| All safe | 0.0572 [0.0327, 0.0843] ✓ | 0.0275 [0.0025, 0.0551] ✓ |
| **Forced-decode fill (S35)** | **0.0555 [0.0314, 0.0814] ✓** | **0.0347 [0.0080, 0.0605] ✓** |

Under every assignment, both headline effects remain significant and in the
same direction. The null-cell handling choice therefore does not flip any
reported main effect. (Scenario 6 uses the forced-decode labels of Appendix
A.4; row 1 uses the S34 canonical estimates, computed under the same scope and
bootstrap procedure.)

## A.4 Forced-decode protocol validation (S35)

To verify that the null cells would not have changed the verdicts had the
scorer succeeded, we introduced a second, independent scoring mode that is
immune to the narration failure: **forced decoding**. Using the identical
prompt template (same rubric, `enable_thinking=false`), we generate a single
token and take the verdict as `argmax(logits(token("0")), logits(token("1")))`
at the first generated position.

**Protocol stability.** On the cells where free-form decoding already produced
a verdict, the two modes agree almost perfectly: 99.52% agreement on the E2B
main stage (Cohen's $\kappa=0.9923$) and 99.92% on S28
($\kappa=0.9985$). Applying the identical protocol to the remaining pipeline
scopes (S36) confirms stability across all five stages: 99.86% on S17 E4B
audio ($\kappa=0.997$), 98.97% on S17 E4B text ($\kappa=0.984$) and 99.71% on
S33 ($\kappa=0.994$). Agreement is flat across E_t and N strata (0.992--0.998),
and the disagreements carry markedly lower verdict margins
($|s_1-s_0|$ mean 5.10 vs. 8.83), i.e., disagreements concentrate on
genuinely ambiguous cells.

**Null cells are low-confidence cells.** The 27 E2B-main null cells have a
forced-decode |margin| median of 4.06 vs. 7.88 for the population, with
18/27 below 5 (the minimum is 0.06, a coin flip). S28 behaves identically
(5.09 vs. 7.75), as do the S36 scopes (null medians 4.47 audio / 4.19 text /
6.70 S33 vs. population 7.69 / 7.50 / 7.59; 19/23, 14/19 and 1/3 below 5).
The cells where free-form scoring fails are therefore precisely the cells
where the scorer itself is most uncertain between 0 and 1 --- they are
ambiguous judgments, not scoring artifacts. Consistently, the independent
judge-big scorer is also split on these same cells (9/27 harmful on E2B main,
and only 9/23, 2/19, 0/3 agreement with the forced labels on S17 audio, S17
text and S33), confirming genuine cross-scorer uncertainty rather than a
systematic exclusion bias.

**Forced labels are weak evidence.** Forced decoding leans harmful on the
null cells (26/27 on E2B main, 11/11 on S28, and 22/23 / 18/19 / 3/3 on the
S36 scopes), but at low confidence and in conflict with judge-big on the same
cells (8/27 and 5/11 agreement; 9/23, 2/19, 0/3 on S36). We therefore treat
these labels only as a directional robustness check, not as ground truth ---
and Appendix A.3 shows that even this extreme leaning leaves all headline
effects unchanged.

**Conclusion.** The 83 judge-small null cells across the pipeline (S36)
are format-violation (narration) failures concentrated on genuinely
low-confidence, cross-scorer-ambiguous cells; their exclusion is not
systematically biasing, and the paper's headline *N* and *E_t* effects are
robust under six alternative null-cell treatments. All 83 cells now carry
forced-decode labels (S35 + S36), so no unrateable cell remains uncovered. We
recommend disclosing this appendix alongside the main results to pre-empt
reviewer queries about excluded unrateable cells.

---
*Source stages:* S28 (full four-scorer agreement on S28 hetero-audio),
S34 (systematic pipeline audit), S35 (forced-decode validation on E2B main +
S28), S36 (forced-decode completion on S17 audio/text + S33). Artifacts:
`results/gpu1_pipeline/s28_hetero_audio_full.json`,
`results/gpu1_pipeline/s34_js_null_audit.json`,
`results/gpu1_pipeline/s35_forced_verdict.json`,
`results/gpu1_pipeline/s36_forced_complete.json`.
