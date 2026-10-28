# InfAnFace constant-mean-shape control — 28 October 2026

## Result and interpretation

**The frozen CNN outperformed the constant control on this training cohort, modestly.** On the exact original 32 official training examples and ground-truth crops, input-pixel error was **4.5561 for the CNN versus 4.8268 for the constant shape**, an absolute reduction of **0.2707 pixels / 5.61%**. The CNN had lower per-image Euclidean error on **19/32 images**, and the control on **13/32**. No ties. This is descriptive **same-training-cohort fit**, not held-out, patient/video-independent, benchmark or clinical accuracy.

One real comparison calculation completed with **zero training updates**. No model initialization was used as a replacement, checkpoint was changed, epoch selected, sample added/removed, crop changed, parameter swept, or official test image opened. The constant is fitted and evaluated on the same targets; the CNN was also trained on these targets. This deliberately optimistic comparison does not show useful generalization, that correct image-to-target correspondence explains the gain, or that a detector-backed pipeline works. The constant control's substantial fit also shows why the original large improvement from random initialization alone was insufficient evidence.

## Protocol, inspection and frozen evidence

Read AGENTS, README, the original research plan, original pilot protocol/report, private configuration, selection, coverage, summary/provenance and completion records before implementation. The checkout was clean at `39a20382709542324b04d0fa2ee5842876cb886c`, branch `research/infanface-e0-audit`, matching its locally recorded upstream after the user's push. No remote refresh, push, upload or commit occurred during this phase. The current changes remain local and uncommitted.

The [comparison protocol](infanface-mean-control-protocol-2026-10-28.md) and `configs/experiments/infanface_mean_control.json` were written before any real-data result calculation. The protocol includes a **pre-results implementation correction** described below; no tolerance, cohort, annotation, mask, preprocessing or checkpoint was changed after results.

The reference is the completed ignored `runs/infanface-train-pilot-2026-10-28/`. Cross-checked **49 historically recorded digests** covering configuration, selection, coverage, original protocol/code/configuration/lock files, alignment review, prior software verification and fixed-final checkpoint. All **15 original run files** were additionally snapshotted before calculation and verified unchanged afterward. Files such as summary/status lacking their own previously recorded digest were frozen now and checked for cross-record consistency/reproduction; this is not invented retrospective authenticity. All **105 pre-milestone historical files** in the preservation inventory (raw data, inventories, original plan, prior reports/private evidence, environments/locks and run artifacts) retained their hashes after completion.

Reconstructed samples by **exact ordered reference join**, without reranking/reselecting, replacement or new identities. Verified official training membership before pixels, source row, annotation digest, source size, EXIF tag, recorded-box equality to the original rule, every separate mask count and eligible-point count. Applied each **recorded box**, not a new chosen box. The original run retained mask counts rather than mask tensors; for this particular all-valid 68/68 cohort, those counts uniquely fix every geometric mask bit. The comparison also stores new target/mask/inverse digests privately. No claim of a pre-existing tensor digest is made.

The shared index skips test rows before numerical label interpretation. Only the same 32 training image members were opened. Archive checksum verification reads compressed container bytes and does not decode test payloads. Source assumptions remain **UNVERIFIED**: base-0 pixel centres, stored-pixel annotation frame, +0.5 to working edges, stored/stored EXIF handling. Tag 0 would not trigger a silent rotation/override; no selected raster has invalid EXIF. Annotation-assisted boxes retain the original 20% span expansion, rounding and clipping of box edges only. Preprocessing remains uniform BICUBIC affine resize/padding to 32×32, RGB CHW float32 `[0,1]`, no augmentation or fitted image normalization. Original annotations and official partitions are unchanged.

## Constant estimator and matching

`models/mean_shape.py` accepts only normalized targets and boolean masks, **never images/features**. For landmark slot j, average x and y over its eligible training observations in float64 using the original float32 target numbers. `predict(N)` repeats that identical normalized shape for every image. This arithmetic mean minimizes point-weighted normalized-coordinate **MSE among constant predictors on the fitted observations**. It does not necessarily minimize Euclidean or image-macro error. No optimization steps or independent fitted parameters beyond these means are used.

Missing observations do not enter the sum/denominator. A zero-support slot stays NaN in tensors and null in saved JSON, with support 0; never a fabricated zero coordinate. Eligible nonfinite targets or CNN predictions cause failure rather than silent attrition. Comparison eligibility is the original mask intersected with nonzero control support; this phase rejects any shrinkage from the original usable observations.

| Denominator / coverage | Both methods |
|---|---:|
| Same fitted/evaluated official training images | 32 |
| Eligible images / selected images | 32 / 32 |
| Matched eligible / original eligible / selected point slots | 2,176 / 2,176 / 2,176 |
| Coordinate scalars for MSE | 4,352 |
| Source landmark slots | 68 |
| Per-landmark support minimum / maximum | 32 / 32 |
| Zero-support landmarks | 0 |
| Newly excluded images/points | 0 / 0 |
| Undefined anatomical-NME denominators | 32 images per method |
| Test image members opened / training updates | 0 / 0 |

Geometric eligibility is not visibility or verified per-point anatomical accuracy. Source identities, source-video mappings and patient independence remain unresolved; no project/validation split was created.

## CNN reproduction and aggregate comparison

Loaded only the digest-verified original `final.pt`, fixed epoch 20, using the existing CPU float32 model in evaluation/no-grad mode. Same original order, batch 8, one thread, deterministic algorithms and zero workers. The existing checkpoint helper reconstructs the stored optimizer state but no optimizer step or backward operation is performed. Synthetic safeguards explicitly reject either operation. No learned checkpoint was written into the comparison directory.

Predeclared reproduction criterion: `abs(new-old) <= atol + rtol*abs(old)`, with MSE `atol=1e-7`, pixel errors `atol=1e-6`, and `rtol=1e-6`. The original evaluator reproduced **all original metrics exactly**, including MSE `0.013662991812452674`. An independent float64 accumulation gives CNN MSE `0.01366299210178018`, a difference of **2.893275063975409e-10**, within the predeclared tolerance. Input and source pixel errors match exactly. Counts/coverage match; discrepancy lists are empty. No tolerance was widened and no result substituted.

MSE below averages eligible coordinate scalars globally. Euclidean errors average points within each image, then give each of the same 32 images equal weight. Input units are 32×32 pixel-edge units; source errors apply each original inverse with no clamping or new half-pixel adjustment, then compare against original source annotation numbers. Source error is resolution/crop-scale dependent. All slots 0–67 are array indices; no anatomical mapping or NME denominator is invented.

| Same-training-cohort metric | Constant control | Frozen CNN | CNN reduction | Relative reduction |
|---|---:|---:|---:|---:|
| Normalized-coordinate MSE, float64 evaluation | 0.015497068270144174 | 0.01366299210178018 | 0.0018340761683639933 | 11.8350% |
| Image-macro input-pixel Euclidean error | 4.826836362962258 | 4.556128670173429 | 0.2707076927888288 | 5.6084% |
| Image-macro source-pixel Euclidean error | 87.2972359838528 | 82.98714856711695 | 4.310087416735854 | 4.9373% |

Relative reduction is `(control-CNN)/control`. A zero control denominator produces null/undefined, including 0/0; it is never reported as zero improvement or infinity. All aggregate and per-image relative denominators were positive here; zero undefined relative comparisons. NME remains undefined separately.

## Descriptive paired per-image differences

Signed differences are **CNN − control**, so negative favors the CNN. Each row uses 32 matched images, 68 points / 136 coordinate scalars per image. Per-image predictions, means, references and differences remain private. Tie tolerance is absolute `1e-12`; no independent-patient assumption, p-value or confidence interval is used.

| Metric | Mean difference | Median | Minimum | Maximum | CNN lower / control lower / ties |
|---|---:|---:|---:|---:|---:|
| Normalized-coordinate MSE | -0.001834076168364 | -0.000322519700324 | -0.015024723028305 | 0.004457559117897 | 18 / 14 / 0 |
| Input-pixel Euclidean error | -0.270707692789 | -0.167455210913 | -1.895435335753 | 1.186355615131 | 19 / 13 / 0 |
| Source-pixel Euclidean error | -4.310087416736 | -1.662847108192 | -60.051083332106 | 20.616016815834 | 19 / 13 / 0 |

The gain is not uniform: the control is better on 13 images by Euclidean error and 14 by MSE. The mean shape is a useful simple reference for these oracle crops, not a deployable localization result. No sample-level results or anatomical judgments are published.

## Software checks, environment and preflight failure

Reused `.venv-baseline`: CPython 3.13.3, Torch 2.14.1, Pillow 13.0.0, macOS 27.0.1 arm64; pytest 8.4.2 / Ruff 0.14.2. **No dependencies or environment/lock files changed.** All 19 pinned distributions plus the editable project matched the existing verified lock inventory. The optional NumPy-bridge warning remains; this path does not use it.

Before the actual result calculation: **310 synthetic tests passed**, Ruff lint and formatting passed (**45 Python files**), both environments' dependency checks passed, and the synthetic metadata/group audit returned 0 with its unknown-identity limitation. The preserved `.venv` passed **250 tests with four expected Torch-module skips**. The 22 added tests cover hand-calculated unequal-mask means/MSE/Euclidean errors, all-missing landmarks and strict JSON nulls, coordinate inversion, zero denominators, paired arithmetic, reproduction tolerances/counts, changed crop/annotation/membership/checkpoint/config/status rejection, non-cohort access rejection before pixels, output constraints, no training updates, immutable originals and discrepancy retention. No infant data is embedded or required.

Development checks initially found a test inverse-dtype mistake and a long error-message line; both were corrected before the final gate. The first real command attempt then **failed at environment preflight, before image loading, CNN inference or control fitting**. Literal `pip freeze` text had changed only this editable repository's Git revision from `a2c682e` to `39a2038` after the user's commit. The 19 pinned dependency lines and all historical source hashes were unchanged. Preserved failed `runs/infanface-mean-control-2026-10-28/` and documented an explicit pre-results protocol correction: allow only that same editable repository/egg's commit field to differ, while checking all other inventory text and historical source/lock hashes strictly. A regression test rejects dependency version, repository, extra-package and unrelated editable-package changes. No new assets/dependencies, metric tolerance, settings or cohort change was involved.

After the correction and a fresh passing 310-test gate, **one actual calculation** completed in `runs/infanface-mean-control-2026-10-28-verified/` at `2026-10-28T19:05:56.604206+00:00`. This is one completed comparison preceded by a failed preflight, not two measured comparisons. There was no real reproduction discrepancy, extra ablation or retraining.

## Commands and private artifacts

Actual final commands from the project root:

```sh
.venv-baseline/bin/python scripts/verify_infanface_pilot_checks.py --output reports/private/infanface-mean-control-software-verification-2026-10-28-final.json
.venv-baseline/bin/python scripts/run_infanface_mean_control.py --config configs/experiments/infanface_mean_control.json --run-dir runs/infanface-mean-control-2026-10-28-verified --checks reports/private/infanface-mean-control-software-verification-2026-10-28-final.json
```

These document the completed comparison, not permission for another result calculation. The command rejects existing destinations, original-run/parent destinations, nonfixed configurations, failed/current-source-mismatched software records, changed frozen inputs and reproduction discrepancies. Failure outputs remain private; do not delete status or change settings to bypass safeguards. For ordinary offline software verification, use the README's development commands; do not rerun the comparison during testing.

The completed ignored directory contains resolved config, frozen original-file/cohort/checkpoint references, provenance/code/lock/protocol/environment hashes and text, environment comparison, CNN reproduction/tolerance details, target/mask/inverse digest, private learned means/support/predictions/per-image differences, aggregate metrics and completion status. The final output-integrity manifest hashes the saved artifacts, including the current environment text. Failed preflight and intermediate/final software check records remain ignored and preserved. No image, selected reference, mean coordinate, prediction, checkpoint or per-image result is Git tracked.

New files: `models/mean_shape.py`, `evaluation/mean_shape.py`, `engine/mean_shape_control.py`, `scripts/run_infanface_mean_control.py`, `configs/experiments/infanface_mean_control.json`, two synthetic test modules, protocol and this report. README and AGENTS updated; original research plan, folder structure, pilot code, historical reports and run outputs unchanged.

## Next bounded proposal — not executed

Propose **one frozen-checkpoint image-correspondence control** on this exact cohort: predeclare a single deterministic nonidentity cyclic permutation of the prepared input tensors, keep each target/mask/inverse in its original position, and compare frozen CNN error against the already recorded correctly paired fit. No retraining, new images, alternative permutations, sample selection, checkpoint selection or test access. This would ask whether correct image-to-target pairing contributes to this training fit beyond the common crop-relative geometry. It still cannot establish generalization, and annotation-derived crop leakage remains. Write/approve its protocol separately before execution.

Before held-out research, source-coordinate/frame and anatomical conventions need stronger evidence, and subject/video grouping and defensible validation rules remain unresolved. Single-cohort fitting, low resolution, direct regression, qualitative alignment review, no detector or visibility labels, target-derived crops and uncertain source rights remain material limitations. This comparison supplies no seizure supervision or clinical evidence. **Stop after this comparison.**
