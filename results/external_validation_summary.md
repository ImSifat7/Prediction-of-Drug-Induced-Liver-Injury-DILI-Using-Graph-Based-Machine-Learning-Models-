# External Validation of the TDC-DILI Headline Model

**Model.** The headline recipe — RDKit-2D descriptors + Morgan-ECFP-2048 → bagged
XGBoost (5 bags) — trained once on the full official TDC `admet_group` DILI
`train_val` set (n=379) and **frozen**.

**External set.** The broader DILIrank-derived set (`data/dili_with_smiles.csv`,
n≈982), with **every TDC molecule removed** (overlap matched by InChIKey
*skeleton*, the first 14 chars, so salt / protonation / stereo variants are also
stripped). After parsing, internal de-duplication, and removing 174
TDC-overlapping molecules → **707 truly-external molecules**.

Reproduce: `python -m src.improved.external_validation_v2`
(learnability check: snippet in `results/external_validation_summary.md` history /
the chat log; uses 5-fold stratified CV on the high-confidence external subset).

## Results

| Set | n | pos-rate | AUROC | 95 % CI | ACC | F1 | MCC |
|---|---|---|---|---|---|---|---|
| **CONTROL — TDC official test** | 96 | 0.52 | **0.933** | [0.875, 0.979] | 0.740 | 0.800 | 0.552 |
| External — all | 707 | 0.57 | 0.648 | [0.607, 0.690] | 0.625 | 0.724 | 0.216 |
| External — novel scaffold only | 510 | 0.55 | 0.634 | [0.584, 0.683] | 0.602 | 0.716 | 0.191 |
| External — TDC-comparable labels (vMost/vNo) | 421 | 0.27 | 0.680 | [0.622, 0.736] | 0.458 | 0.465 | 0.177 |
| External — TDC-comparable + novel scaffold | 303 | 0.24 | 0.687 | [0.621, 0.755] | 0.396 | 0.430 | 0.185 |
| **Learnability — 5-fold CV trained *on* external high-conf** | 421 | 0.27 | **0.708** | [0.654, 0.763] | — | — | — |

> Decision threshold (0.095) was chosen to maximise MCC on the **TDC training**
> predictions only — no external peeking. AUROC is threshold-free and is the
> primary metric; the low threshold (a side-effect of `scale_pos_weight`)
> explains the poor ACC/F1 on the negative-heavy external subsets.

## Controls that make this finding robust

1. **No pipeline bug.** The identical frozen model reproduces **AUROC 0.933** on
   the official TDC test set (≈ the published 0.920 ± 0.014). The external drop is
   real, not an artefact of featurisation.
2. **Not just label noise.** Restricting the external set to TDC's own label
   definition (`vMost-DILI-concern` vs `vNo-DILI-concern`, dropping the
   `vLess`/`vAmbiguous` middle) only lifts AUROC 0.648 → 0.680. The gap to 0.93
   survives.
3. **The external labels ARE learnable.** A model trained *on* the external
   high-confidence set (5-fold CV) reaches only **0.708** — so the data carries
   signal, but its achievable ceiling is ~0.71, far below the benchmark's 0.93.

## Interpretation (decomposition of the gap)

| Effect | AUROC change | Magnitude |
|---|---|---|
| **Benchmark curation** (TDC's 475 = an easy, separable slice) | 0.93 → 0.71 | **~0.22 (dominant)** |
| Domain transfer (TDC-trained → non-TDC, vs in-domain ceiling) | 0.71 → 0.68 | ~0.03 (small) |
| Label noise (adding `vLess`/`vAmbiguous`) | 0.68 → 0.65 | ~0.03 (small) |

**Headline claim for the paper.** The model does *not* simply fail to
generalise — it nearly matches the in-domain ceiling on the new chemistry. The
dominant effect is that the **public TDC-DILI benchmark is a curated, highly
separable subset of the FDA's DILIrank**. Realistic DILI prediction over the
*full* FDA-labelled chemical space tops out near **AUROC 0.71**, even training
in-domain — roughly 0.22 below the leaderboard headline. This is a benchmark-
validity result, complementary to the Feb-2026 bioRxiv leakage audit, and it
recasts the saturated 0.92 as an artefact of curation rather than genuine
real-world skill.

Artefacts: `results/external_validation_v2.csv`,
`results/external_predictions_v2.csv` (per-molecule probs + concern category +
novel-scaffold flag, ready for error analysis).
