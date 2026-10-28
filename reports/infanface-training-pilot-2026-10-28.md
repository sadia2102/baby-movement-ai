# InfAnFace first real-data feasibility pilot — 28 October 2026

## Outcome and scope

**Completed one ground-truth-crop, TRAINING-ONLY optimization smoke test.** The unchanged randomly initialized 75,832-parameter CNN trained on 32 deterministically selected official training images for 20 epochs / 80 Adam updates, CPU only. No held-out data, validation/model selection, project split, infant identity inference, pretrained asset or clinical label was used. The fixed final checkpoint reloaded with exactly equal predictions on all 32 training inputs. This establishes that this software can optimize the selected real-image targets; it does not establish useful localization, image-dependent learning, benchmark accuracy, patient/video independence or seizure detection.

The [fixed protocol](infanface-training-pilot-protocol-2026-10-28.md) and public JSON configuration were written before preparation and real model execution. No parameter sweep, replacement sampling, failed real run or second training run occurred. Preparation and final-checkpoint reload are separate stages of this one experiment, not additional training runs.

## Inspection and software changes

Read AGENTS, README, the original research plan, current acquisition/duplicate evidence, preprocessing/synthetic-baseline reports and environment records before edits. Starting checkout was clean at `a2c682e898a1c8a2af4796341fa331c36b3c5c2a`, branch `research/infanface-e0-audit`, tracking the matching locally recorded remote branch. `origin` remains `https://github.com/sadia2102/baby-movement-ai.git`. No fetch, push, commit, remote change or network operation was performed. Execution records correctly mark the working tree dirty and retain tracked-diff and individual runtime/config hashes, including new untracked source files.

- `datasets/infanface_pilot.py`: training-only shared-index reader and deny-first image loader, deterministic SHA selection, bounded configuration and explicit ground-truth crop utility. Test rows are discarded using official membership **before numeric annotation/attribute parsing**. Only selected training image members can be opened. The existing general reader/loader/preprocessing interfaces are unchanged.
- `engine/infanface_pilot.py`: separate no-model preparation and reviewed execution stages; private immutable selection/config/coverage records; review, checksum and source-hash gates; fixed-final-only training/checkpoint/reload; failure/completion status and rerun refusal. Preserves original annotations, unknown provenance and individual geometric masks.
- `scripts/run_infanface_pilot.py`: explicit `prepare` and `run` commands with no pretrained checkpoint, validation partition or test-data option.
- `scripts/verify_infanface_pilot_checks.py`: offline software checks and a private verification record bound to current source/config/protocol/lock hashes; checks exact installed lock versions before execution.
- Two synthetic test modules and a manufactured-archive fixture in `tests/conftest.py`: 40 new cases. README/AGENTS now distinguish the completed synthetic phase from this narrowly authorized real pilot. Original plan, old reports, locks, environments and raw files are preserved.

## Selection, coordinates and preprocessing

Only the shared CSV index was read; official test numeric labels were not interpreted and no test image payload was opened. Whole-container checksum verification necessarily reads the compressed archive bytes; it is not image/annotation decoding of the test partition. The archive matched its receipt SHA-256 `086efbcdb3e7bc9b718c012678d263f74abe1df42e018bac3574b721ab9d1c57` before and after preparation, before the run's selected-image loading, and after training/reload. No extraction or raw modification occurred.

Rank the 210 validated official training references by SHA-256 of UTF-8 `infanface-train-fit-v1|42|<image-set>/<filename>` (tie-break by image set/name), take the first 32, and never replace excluded examples. Sample references, source rows, annotation digests, selection digest and explicit boxes remain only in the ignored local run. No subject/video/source-group/project identities were created.

Assume **base-0 source pixel centres, stored-pixel annotation frame**, then add 0.5 for the existing working edge frame. Both hypotheses remain **UNVERIFIED**. Prior eight training-only overlays supported gross stored-frame correspondence, including two EXIF-rotated examples; they could not resolve a one-pixel origin difference. This phase reviewed all 32 selected images using two private annotation-only crop panels before model initialization. All supported gross facial-region/landmark correspondence, including one EXIF-6 stored raster. This single assistant qualitative review is neither independent anatomical adjudication nor proof of every landmark's accuracy/global convention. Profile, occlusion, low detail, text and layout content limit interpretation. No selection/exclusion was based on model outputs.

For each selected raster, derive the axis-aligned box from at least three present finite, in-footprint landmarks with nonzero x/y span. Expand each side by 20% of its axis's span, floor left/top, ceil right/bottom and clip **box edges only**. Existing reversible preprocessing uses uniform BICUBIC affine resize and centred black padding to 32×32. RGB tensors are CHW float32 `[0,1]`, without augmentation or fitted normalization; targets are 68 normalized edge pairs `(x/32,y/32)`. Preserve raw coordinates and forward/inverse transforms. This is an annotation-assisted/oracle crop; detection coverage and detector-backed operation were not assessed. The crop also leaks target geometry into input normalization, so improved training fit alone may reflect a generic crop-relative shape.

Use stored orientation consistently for both pixels and coordinates. Do not rotate, normalize or guess an override, even for invalid EXIF metadata. Selected tags were **29 absent, two tag 1, one tag 6; zero invalid tags**. Historically, one invalid tag `0` was reported; it was not specially sought or reopened. Stored/stored handling retains that raster, while requested normalization/frame conversion fails explicitly. Synthetic tests exercise tag 0 preservation. The historical single out-of-bounds landmark belonged to the official test pool and was not reopened or used in any decision.

Eligibility is present AND source/crop/output footprint masks under the declared hypothesis. Finite out-of-bounds coordinates remain numerical and masked, never clamped; missing values remain missing/NaN with false masks. These are geometric eligibility masks, **not visibility labels**. Decode/crop/all-target failures are private exclusions with reason, no replacement; an empty usable cohort stops. This run had no such exclusions or masked slots.

| Coverage | Actual result |
|---|---:|
| Official training metadata pool | 210 |
| Deterministically selected / decoded / usable | 32 / 32 / 32 |
| Excluded images | 0 |
| Eligible / selected point slots | 2,176 / 2,176 (100%) |
| Missing, source-OOB, crop-OOB, output-OOB mask counts | 0 each |
| Official test images opened | 0 |
| Project splits or inferred identities | 0 |

The 100% figure describes selected geometric coverage, not annotation quality, visibility, cohort representativeness or all-training-pool decoding coverage. Only 32 of 210 training images were decoded this phase.

## Fixed optimization and observed training fit

CPU float32, one thread, deterministic algorithms, zero data-loader workers. Fresh random initialization seed 42, shuffle generator seed 43. Adam LR 0.003, betas `(0.9,0.999)`, eps `1e-8`, weight decay 0, batch 8, exactly 20 epochs, four updates per epoch, no scheduler/augmentation/early stopping. No synthetic checkpoint was loaded. Loss is MSE over eligible x/y scalars; all-missing batches skip updates, an all-unsupervised epoch fails, and nonfinite loss/gradient fails before updates. Only the final epoch was saved; no best-epoch selection or held-out validation was performed.

Errors below use all 68 array slots, eligible landmarks averaged within each image and then averaged equally over the same 32 **training images**. Input error is Euclidean distance in 32×32 pixel-edge units; source error maps predictions through each recorded inverse to original source annotation pixel units. Source image/crop sizes vary, so the source-pixel aggregate has a resolution dependence. No anatomical index interpretation, eye denominator, NME, patient weighting or clinical metric is asserted. Without supplied verified denominators, NME remains null with 32 undefined-denominator images, not a substituted normalization.

| Training-set quantity | Fresh initialization | Fixed final epoch 20 |
|---|---:|---:|
| Normalized-coordinate MSE | 0.27231643348932266 | 0.013662991812452674 |
| Image-macro input-pixel Euclidean error | 22.669556831104792 | 4.556128670173429 |
| Image-macro source-pixel Euclidean error | 415.9650109760379 | 82.98714856711695 |

Within-epoch optimization loss was `0.2157815508544445` at epoch 1 and `0.01271415501832962` at epoch 20. These shuffled, changing-model averages differ from fixed-model initial/final evaluation MSE. Every epoch saw 2,176 eligible points with four updates and zero skipped batches. Completion recorded at `2026-10-28T17:56:26.140102+00:00`. Reload yielded identical normalized predictions on all 32 training images (maximum absolute difference 0), and identical aggregate metrics. No performance threshold or scientific success criterion was selected after observing fit.

## Software/environment verification, distinct from fit

Before real preparation/model execution, the existing baseline was rechecked at **248 passing synthetic tests**. After additions, the recorded offline gate passed:

| Check | Actual result |
|---|---|
| `.venv-baseline` full pytest | **288 passed**, no skips |
| Ruff lint / format | Passed / **39 files formatted** |
| Baseline dependency consistency | Passed |
| Baseline lock version inventory | All **19** pinned distributions plus editable project, no additions |
| Preserved `.venv` full pytest | **250 passed, two optional Torch modules skipped** |
| Preserved `.venv` dependency consistency | Passed |
| Synthetic metadata/group audit | Exit 0; its unknown-identity limitation retained |

New tests use manufactured pixels only. They independently check the SHA ranking rule, malformed test payload filtering before numeric parsing, rejected test/forged-batch image access, immutable archive bytes, fixed bounds, crop rounding/OOB preservation, no-model preparation/no replacement, mask arithmetic, invalid EXIF preservation, missing-target update refusal, changed-input/review/software gate rejection, private path/symlink constraints, fixed-final reload and rerun refusal. During development, an initial test collection failed because of pytest's import mode; the manufactured archive fixture was moved into `conftest.py`, then targeted and full checks passed. Initial formatting/lint issues were corrected before the recorded gate. No real-run failure occurred.

Reused **CPython 3.13.3, macOS 27.0.1 arm64, Torch 2.14.1, Pillow 13.0.0**, pytest 8.4.2, Ruff 0.14.2 and the existing hash-locked environment. No dependency, lock, system Python or environment modification. The existing optional NumPy-bridge warning still occurs; this pipeline does not use that bridge. The record verifies installed versions against the existing previously verified wheel lock; it does not claim a fresh package download/hash reinstallation. Raw archive, existing inventory, original plan, old reports/private evidence, historical environment records/locks and previous run artifacts were checked against their pre-milestone hashes after completion.

## Exact commands and private records

Executed from the project root, in this order:

```sh
.venv-baseline/bin/python scripts/verify_infanface_pilot_checks.py --output reports/private/infanface-pilot-software-verification-2026-10-28.json
.venv-baseline/bin/python scripts/run_infanface_pilot.py prepare --config configs/experiments/infanface_train_pilot.json --archive data/raw/infanface/InfAnFace.zip --receipt data/raw/infanface/acquisition-receipt-2026-10-27.json --protocol reports/infanface-training-pilot-protocol-2026-10-28.md --run-dir runs/infanface-train-pilot-2026-10-28
# Review all usable training annotation panels. Only after actual review, create
# alignment-review.json from its private template, record evidence/limitations,
# hashes, reviewer and count; unresolved source conventions stay UNVERIFIED.
.venv-baseline/bin/python scripts/run_infanface_pilot.py run --run-dir runs/infanface-train-pilot-2026-10-28 --checks reports/private/infanface-pilot-software-verification-2026-10-28.json
```

These are historical executed commands, **not authorization for another run**. The completed directory is intentionally not reusable: preparing there or starting again is rejected. An unreviewed preparation cannot proceed. A future authorized reproduction needs a fresh directory, unchanged reviewed protocol/configuration or a separately predeclared protocol, fresh software verification, and an actual review. Do not bypass gates by deleting records or changing completion status.

Ignored `runs/infanface-train-pilot-2026-10-28/` contains config, selection/annotation hashes and unknown provenance fields, coverage, source/code/protocol/archive hashes, QC panels/review, Git/environment/seed records, initial and final training fit, per-epoch losses/coverage, `final.pt` and its digest, exact reload verification and completion status. Software check outputs are ignored under `reports/private/`. Images, boxes, sample lists/digests, learned weights and run records were not added to Git; share only these aggregate findings.

## Next justified experiment and unresolved evidence

Stop after this pilot. The next bounded proposal is a **training-only constant-mean-shape control on this exact frozen cohort and preprocessing**, with a protocol written before execution. Compare its fixed-crop input-pixel fit against this CNN to determine whether the learned result adds anything beyond label-derived crop normalization and a generic shape. This would still be descriptive training fit, not held-out performance. Do not run it under the current authorization or select new samples/settings from these results.

Before any generalization pilot: obtain stronger source-coordinate/frame and anatomical-index/normalization evidence; adjudicate alignment independently; resolve or explicitly limit source/subject grouping using permitted provenance; predeclare a defensible validation/model-selection protocol; review downstream learned-weight/media rights and intended use. Unknown infant/video mappings remain unresolved. Keep official test sealed until a justified final-evaluation protocol is frozen. Low 32×32 resolution, simple direct regression, no detector, qualitative QC, oracle-crop leakage, absent visibility labels, unrepresentative sampling and a single training run remain material weaknesses. No finding here provides seizure supervision, clinical accuracy or diagnostic evidence.
