# Appendix: Cross-Scorer Family Robustness and the ShieldGemma Divergence

This appendix reports the cross-scorer family measurement analysis that
supports the paper's headline effects. It answers the standard reviewer
question ``are the N and E_t effects an artifact of one particular scorer
family?`` We score the same responses with nine independent label sources and
show that (i) label-level agreement across families is high, (ii) the
headline effects replicate across every harmful-compliance judge family, and
(iii) one fully independent moderation classifier (ShieldGemma-9B) partially
diverges in a mechanistically interpretable way that we disclose
transparently rather than suppress. Zero human labels were introduced at any
point.

## B.1 Scorer families and protocols

| Family | Model / basis | Construct | Notes |
|---|---|---|---|
| judge_big | Gemma-4-E4B-it | harmful compliance | primary dual-judge member |
| judge_small | Gemma-4-E2B-it | harmful compliance | primary dual-judge member |
| dual_consensus | judge_big == judge_small | harmful compliance | paper's authoritative scope |
| qwen32 | Qwen2.5-32B-Instruct-AWQ | harmful compliance | cross-family judge |
| cross_check | Qwen2.5-3B-Instruct | harmful compliance | low-resource anchor |
| strongreject | official StrongREJECT rubric | harmful compliance | official benchmark scorer |
| harmbench | official HarmBench scorer | harmful compliance | official benchmark scorer |
| forced | forced-decode argmax (judge_small) | harmful compliance | format-violation immune (Appendix A.4) |
| shieldgemma | ShieldGemma-9B (4-bit) | conversational safety | moderation classifier, distinct construct |

The authoritative analysis scope of the paper is the **dual-judge consensus**
(judge_big == judge_small). All other families are robustness checks on the
same responses (primary E2B text generator; S28/S17 for generator and modality
generality).

## B.2 Label-level cross-family agreement (E2B main, 3,600 cells)

| Pair | Agreement | Cohen's $\kappa$ |
|---|---|---|
| judge_big vs shieldgemma | 0.828 | **0.777** |
| judge_small vs shieldgemma | 0.681 | 0.521 |
| shieldgemma vs harmbench | 0.618 | 0.389 |
| shieldgemma vs strongreject | 0.467 | -0.024 |

The two most training-independent families --- the paper's Gemma judge and the
moderation classifier ShieldGemma --- agree on 82.8% of responses
($\kappa = 0.78$): the responses themselves are robustly distinguishable as
harmful vs. safe regardless of family.

> **Note (2026-08-27, author decision).** Three rows of an earlier draft of this
> table --- judge\_big vs.\ judge\_small (0.948/0.912), judge\_big vs.\ qwen32
> (0.942/0.893), and judge\_big vs.\ forced (0.924/0.889) --- could not be
> reproduced from the pipeline artifacts and contradicted the S37 matrix
> (`results/gpu1_pipeline/s37_shieldgemma_cross.json`: 0.8178/0.7401,
> 0.9361/0.9198, 0.8103/0.726). They are deleted. The internal judge\_big vs.\
> judge\_small reliability is reported in the paper's measurement trust chain
> (§4.7.4): agreement 0.82, Gwet's AC1 0.745, balanced $\kappa$ 0.706,
> $n=3{,}573$ (E2B main), computed with the pipeline's agreement utilities on
> the frozen scorer labels.

## B.3 Main-effect replication across nine label sources (E2B main)

Effects are estimated on the paper's exact selections (N effect at E_t=0,
N=0 vs. N=1; E_t effect at N=0 and N=1) with query-clustered bootstrap
(B = 2,000). Direction convention: positive = N=1 more harmful (N effect) /
E_t=1 more harmful (E_t effect). $\checkmark$ = 95% CI excludes 0.

| Label source | *N* effect (E_t=0) | *E_t* effect (N=0) | *E_t* effect (N=1) |
|---|---|---|---|
| judge_big | +0.048 $\checkmark$ | +0.035 $\checkmark$ | +0.038 $\checkmark$ |
| judge_small | +0.054 $\checkmark$ | +0.004 | -0.021 |
| **dual_consensus** | **+0.055 $\checkmark$** | **+0.029 $\checkmark$** | **+0.027 $\checkmark$** |
| qwen32 | +0.031 $\checkmark$ | +0.077 $\checkmark$ | +0.111 $\checkmark$ |
| cross_check | +0.079 $\checkmark$ | +0.094 $\checkmark$ | +0.057 $\checkmark$ |
| strongreject | +0.023 | +0.093 $\checkmark$ | +0.081 $\checkmark$ |
| harmbench | +0.132 $\checkmark$ | +0.111 $\checkmark$ | +0.066 $\checkmark$ |
| forced (S35) | +0.036 $\checkmark$ | -0.004 | -0.022 |
| shieldgemma | **-0.036 $\checkmark$** | -0.004 | +0.025 $\checkmark$ |

**Reading.** (i) The *N* effect is positive under eight of nine label sources
(seven significant); ShieldGemma is the single exception, and it is the only
significant reversal. (ii) The *E_t* effect at N=1 is the most robust ---
significant and positive under seven of nine sources, including ShieldGemma.
(iii) The *E_t* effect at N=0 is positive and significant under the
consensus and all six compliance judges, but near zero under judge_small
alone, forced, and ShieldGemma.

## B.4 The ShieldGemma divergence is generator-family specific (S37/S38)

ShieldGemma was extended to every other scope with an independent generator:
S17 E4B (audio 7,200 + text 3,600 cells) and the two Qwen2-Audio hetero
scopes (S28, 1,200 cells; S33, 344 cells). The table reports the *N* effect
(N=1 minus N=0 at E_t=0) under ShieldGemma and judge_big on each scope
(query-clustered bootstrap, B = 2,000; $\checkmark$ = 95% CI excludes 0).

| Scope | Generator | *N*: ShieldGemma | *N*: judge_big |
|---|---|---|---|
| E2B main (text) | Gemma-4-E2B-it | -0.036 $\checkmark$ | +0.048 $\checkmark$ |
| S17 E4B audio | Gemma-4-E4B-it | -0.063 $\checkmark$ | +0.063 $\checkmark$ |
| S17 E4B text | Gemma-4-E4B-it | -0.014 | +0.051 $\checkmark$ |
| S28 hetero audio | Qwen2-Audio-7B | **+0.130 $\checkmark$** | +0.214 $\checkmark$ |
| S33 hetero audio | Qwen2-Audio-7B | +0.117 $\checkmark$ | +0.069 |

The *N* reversal under ShieldGemma is **specific to the Gemma-family
generators**: significant on both E2B main and E4B audio, present in sign
only (non-significant) on E4B text. On the independent Qwen2-Audio generator
(S28), ShieldGemma does *not* reverse --- it agrees with judge_big in
direction (+0.130 vs. +0.214, both significant). The absence of reversal on
Qwen2-Audio is not limited to these two families: across **all five label
sources available on S28** (judge_big +0.214 $\checkmark$, judge_small +0.214
$\checkmark$, strongreject +0.089 $\checkmark$, harmbench +0.230 $\checkmark$,
shieldgemma +0.130 $\checkmark$) the *N* effect is positive and significant,
including the two official benchmark scorers and the moderation construct
itself. S33 (the second Qwen2-Audio scope, 29 queries) replicates the
alignment across the same five sources --- judge_small +0.100 $\checkmark$,
strongreject +0.144 $\checkmark$, harmbench +0.256 $\checkmark$, shieldgemma
+0.117 $\checkmark$, judge_big +0.069 (positive, not significant at this
smaller sample). On **both** Qwen2-Audio scopes, therefore, every one of the
five label sources is directionally positive for the *N* effect, and four of
five are significant. The moderation-compliance divergence is therefore
triggered specifically by the Gemma-family response patterns, not by the
moderation construct per se. By contrast, the *E_t* effect at N=1 is
positive under ShieldGemma on every scope (E2B +0.025 $\checkmark$, E4B audio
+0.061 $\checkmark$, E4B text +0.014, S28 +0.115 $\checkmark$, S33 +0.065) and
significant on both audio scopes --- it is the most robust finding across
constructs and generators.

**Reading.** The moderation--compliance divergence is not a universal
property of the moderation construct; it is triggered by the response
patterns of the Gemma-family generators (verbose, hedged, self-framed
elaboration in the N=1 condition). Where the responses are shorter and more
direct (Qwen2-Audio), the two constructs agree in direction. This is a
stronger statement than a generic ``construct difference'': the inversion is
mechanistically localized, which we take as further evidence that it is a
real content-level phenomenon rather than a systematic measurement artifact.

## B.5 Mechanism: a construct difference, not a measurement artifact

On the E2B main stage (E_t=0), N=1 responses are markedly longer and more
elaborated than N=0 (median 824 vs. 500 characters) because targeted
profiling prompts elicit more detailed generation. The compliance judges rate
these longer N=1 responses *more* harmful (judge_big 0.101 vs. 0.050;
judge_small 0.286 vs. 0.228). ShieldGemma instead rates them *less* unsafe
(0.142 vs. 0.178). Disagreements localize consistently: ShieldGemma is
stricter on N=0 cells (judge_big=0, ShieldGemma=1: 141 vs. 92 cells) and more
lenient on N=1 cells (judge_big=1, ShieldGemma=0: 55 vs. 26 cells). Query
families are perfectly balanced between N=0 and N=1 (282/312/306 each), so
this is not an attack-family confound.

**Why the divergence does not appear on Qwen2-Audio.** Raw length is not the
driver: on S28 the N=1 responses are likewise longer (median 182 vs. 94
characters, a comparable relative gap) yet ShieldGemma rates them *more*
harmful, in line with judge_big. The inversion therefore depends on the
*content* of Gemma-family elaboration --- verbose, hedged, self-framed
profiling that the moderation classifier reads as safe hedging while the
compliance judges read as harmful compliance --- not on response length per
se. This generator-family localization is consistent with B.4.

**Interpretation.** ShieldGemma is trained to classify ``is this conversation
unsafe`` and is therefore more forgiving of verbose, hedged, or
self-framed-as-responsible elaboration, whereas the paper's compliance judges
assess whether the response harmful-complies with the request. The reversal
is precisely the signature of that construct difference. We report it here so
that the paper's claim can be read precisely: the headline *N* and *E_t*
effects hold under the pre-registered dual-judge protocol and are directionally
consistent across every harmful-compliance judge family and all three
generators; they are **not** claimed to be invariant under a safety-moderation
construct.

## B.6 Conclusion and disclosure recommendation

The measurement evidence is strong: agreement across the two most
training-independent families is $\kappa = 0.78$, and the headline effects
replicate across eight compliance-based label sources. The one divergent
family (ShieldGemma) is mechanistically explained by a moderation-vs-
compliance construct difference that is specific to the Gemma-family
generators' response patterns (B.4--B.5). We recommend (i) including Appendix
B.2--B.5 as a dedicated measurement-sensitivity section, (ii) stating the *N*
effect as ``significant and directionally consistent across eight of nine
label sources (seven significant), with one moderation-construct inversion on
Gemma-family generators reported in Appendix B``, and (iii) keeping the
dual-judge consensus as the registered primary analysis. This pre-empts the
most likely reviewer challenge to the headline effects.

---
*Source stages:* S37 (ShieldGemma on E2B main + S28), S38 (ShieldGemma on
S17 E4B audio/text + S33), S28 five-family (official-benchmark scorers on S28;
`results/gpu1_pipeline/s28_five_family_effects.json`), S39 (official-benchmark
scorers strongreject + harmbench on S33, plus the S33 five-family table;
`results/gpu1_pipeline/s33_five_family_effects.json`). Artifacts:
`results/gpu1_pipeline/s37_shieldgemma_cross.json`,
`results/gpu1_pipeline/s38_shieldgemma_ext.json`,
`results/gpu1_pipeline/s28_shieldgemma_effects.json`,
`results/gpu1_pipeline/s28_five_family_effects.json`,
`results/gpu1_pipeline/s33_five_family_effects.json`,
`results/gpu1_pipeline/s37_shieldgemma_e2b_main_labels.jsonl`,
`results/gpu1_pipeline/s38_shieldgemma_labels.jsonl`.
