# Minimal landmark baseline and synthetic training verification

**Date:** 28 October 2026. **Starting state:** clean branch `research/infanface-e0-audit`, commit `3124719`, tracking its configured origin branch. Read repository rules, README, research plan, preprocessing report/interfaces and environment records before editing. Re-ran all 209 pre-existing synthetic tests, lint, formatting and dependency checks successfully before changes.

## Scope and result

Implemented a minimal randomly initialized 68-point coordinate baseline, tensor adapter, ordinary CPU training/validation loop, masked MSE, independently checked source-coordinate metrics, checkpoint loading and private run records. Created a separate hash-locked CPU environment. A private annotation-only review inspected eight official training examples. No real infant image entered model training or model prediction, and no official test image was opened for inspection.

**248 synthetic tests passed**, including the previous 209 plus 39 new metric/baseline checks. Both environment dependency checks pass; lint and all 33 Python file format checks pass. The original `.venv` passes 224 tests with one optional Torch test module skipped. No unresolved synthetic test failure remains.

The fixed synthetic experiment passed its predeclared engineering acceptance rule:

| Item | Observed result |
|---|---|
| Manufactured training / validation images | 32 / 16, each 32×32 |
| Model parameters | 75,832, all randomly initialized |
| Epochs / optimizer steps | 120 / 480 |
| Best checkpoint selection | Lowest validation normalized-coordinate MSE, epoch 119 |
| Initial / selected validation MSE | 0.2676038146018982 / 0.000034110201340809 |
| Initial / selected image-macro landmark error | 23.00578460050085 / 0.22398287668670216 synthetic source pixels |
| Validation coverage | 16/16 images, 1,088/1,088 landmark slots |
| Reload comparison | All 16 validation prediction arrays exactly equal; max difference 0 |
| Acceptance rule, fixed before first run | Error <1 pixel and <10% of initial; exact checkpoint reload |
| Real images used in model training or prediction | 0 |

This task shows moving vertical/horizontal Gaussian stripes whose intersection translates a fixed 68-point ellipse. It tests spatial coordinate regression and engineering plumbing. It contains no infant anatomy, clinical events or independently labeled real images. The validation images are manufactured with seed 43 rather than the training seed 42; validation also selects the checkpoint, so this is not an independent final test or an InfAnFace result.

An initial identical configuration was retained. After expanding checkpoint verification from one probe to the entire validation set and recording additional contract metadata, the same fixed configuration was repeated without changing learning settings. Both runs produced the same validation metrics and best-checkpoint digest. This bounded repeat is not a parameter sweep or proof of cross-platform determinism.

## Coordinate evidence and bounded training-only review

Existing retained author documentation describes landmark coordinates but does not establish a zero/one-based centre origin or stored/display raster convention. Prior reports already record seven EXIF rotations, one invalid orientation tag `0`, and one out-of-bounds landmark. Those records were preserved, including the prior test-partition quality finding; the current phase did not re-open its test image.

Selection was fixed from validated metadata and existing private audit evidence: first two examples in each training collection batch (`ads`, `google2`, `youtube2`), plus first two additional training examples with EXIF orientation 6 or 8. Budget was **eight examples**, with zero test images selected. The selected tag distribution was absent 5, tag 1 in 1, tag 6 in 1, tag 8 in 1. Four annotation-overlay hypotheses were compared: base-0/stored/stored, base-1/stored/stored, base-0/stored/normalize, and base-0/display/normalize. No model, detector, facial recognition or infant identity assignment was used. The review ran in the original environment without Torch.

All eight show gross annotation/face correspondence under the stored-raster hypothesis. In the two rotated cases, normalizing pixels and stored-frame coordinates together maintains correspondence; assuming the raw annotations were already display-frame coordinates visibly misaligns them. This provides **tentative example-specific support** for stored-frame interpretation. Base-0 versus base-1 differences could not be adjudicated at the review scale. It is a single assistant qualitative review of a deterministic convenience sample, not independent anatomical adjudication. **Source origin, global annotation frame and canonical index semantics remain UNVERIFIED.** No convention was promoted to verified ground truth.

Private panels, selection and decisions remain under `reports/private/baseline-alignment-review-2026-10-28/`; none is embedded in this report. Existing raw data and annotations remain unchanged.

**Invalid EXIF handling:** callers explicitly declare stored or normalized processing. Stored pixels with stored-frame annotations keep the original raster even for tag `0`; the tag/assumption is retained in adapter metadata. Requested normalization/frame conversion rejects invalid tags, with no silent override or guessed orientation. A real pilot needs a predeclared, evidence-supported handling decision; no permanent image exclusion was made here.

**Out-of-bounds handling:** retain original numbers and transformed targets, with separate masks. The adapter's default eligibility rule is present AND source/crop/output footprint bounds. It excludes an out-of-bounds slot from loss without clamping it, zero-filling it or discarding the whole sample. This is a geometric policy, not evidence of occlusion/annotation invalidity; any real-data application still needs explicit protocol approval. No current model consumed that real annotation.

## Input, targets and model

`datasets/torch_landmarks.py` wraps the existing verified ZIP context with `InfAnFaceTensorDataset`, requiring an explicit official partition and `LandmarkInputConfig`. The original sample objects remain accessible on `.samples`, with unknown identities/source groups/project splits untouched. `num_workers=0` is enforced because the archive belongs to its open context; iteration after closing fails. The loader and preprocessing interfaces were not rewritten.

Each item returns contiguous RGB **CHW float32 `[0,1]`**, 68 normalized **pixel-edge `(x/W,y/H)`** target pairs, raw original coordinates as float64, forward/inverse matrices, separate present/source/crop/output masks and their combined loss eligibility, official partition/source-row references and declared origin/frame/EXIF metadata. Explicit missing coordinates remain NaN plus a false mask. Present out-of-bounds values stay numerical and un-clamped. Geometric masks do not establish per-landmark visibility. The source reader's finite 68-pair CSV contract is unchanged.

`decode_coordinates` scales normalized edges by output width/height, then applies the recorded inverse in float64 to recover original annotation numbers. It does not add a second half-pixel offset or clamp outside predictions. Batched/non-square inverse arithmetic and the full manufactured ZIP→preprocessing→tensor→optimizer path are tested. Real adapter samples were not run through the model.

`SmallLandmarkCNN` takes **B×3×32×32** and returns **B×68×2**. Two strided 3×3 convolutions (8 then 16 channels), ReLU, spatial flattening, a 64-unit hidden layer and a linear head preserve coarse position. Direct regression avoids heatmap target/stride/argmax machinery for this engineering milestone. There is no sigmoid output clamp, pretrained state, face detector, augmentation or implicit download. The original plan's HRNet experiment remains a later proposed experiment.

Weaknesses include low image resolution, no spatial uncertainty/confidence head, simplistic manufactured targets, and the tendency of coordinate regression to average ambiguous locations. It establishes no real facial-landmark accuracy, expression recognition, temporal movement accuracy or clinical validity.

## Training, validation and checkpoints

The public JSON config fixes seed 42, batch 8, Adam LR 0.003, 120 epochs, 32 training and 16 validation examples, CPU. The runner uses float32, one CPU thread, zero loader workers and deterministic algorithms. The manufactured training-loader generator uses seed 44. Config validation caps epochs at 200 and sample counts at 64 and rejects non-CPU execution in this milestone. No scheduler, mixed precision, TorchVision pipeline, accelerator fallback or model download is required.

Masked MSE is the mean squared error over eligible x/y scalars. Boolean indexing happens before subtraction, so excluded target NaNs cannot contaminate gradients. Loss aggregation weights batches by eligible landmark count. An all-missing batch returns no loss and takes no optimizer step; it is counted as skipped. Entirely unsupervised train/validation epochs and empty loaders fail instead of claiming zero loss or selecting a bogus checkpoint. Nonfinite predictions/eligible targets fail rather than becoming excluded favorable errors. These cases have explicit synthetic tests.

The ordinary `run_epoch` and `fit` functions accept explicit loaders. The runnable command `verify_synthetic_baseline.py` constructs only manufactured data and exposes **no archive/real-data option**. No real-data training CLI was introduced. Validation runs without gradients; the lowest validation normalized-coordinate MSE selects `best.pt`, and `last.pt` preserves the final state.

Each new ignored `runs/` directory records config, data recipe/content digests, source/config/lock hashes, Git SHA/branch/dirty-diff hash, exact environment, device/seeds, epoch logs and coverage, best/last checkpoints, checkpoint SHA-256, summary and completion/failure status. Checkpoints contain model/Adam state dictionaries, config/epoch and Python/Torch/loader RNG states; modules are not pickled. The loader checks a supplied local digest and size before `torch.load(..., map_location="cpu", weights_only=True)`. Model/optimizer and optional RNG restoration are tested. Exact interrupted-training resume is not implemented or claimed. Use only locally generated checkpoints. See [official checkpoint loading documentation](https://docs.pytorch.org/docs/2.14/generated/torch.load.html).

The completed final run is `runs/synthetic-landmark-baseline-2026-10-28-verified/`; earlier outputs were preserved. All checkpoints/run outputs remain local and Git ignored. Reproducibility is limited to the tested environment; [PyTorch's guidance](https://docs.pytorch.org/docs/2.14/notes/randomness.html) explains version/platform limits.

## Independently checked metrics

`evaluation/landmark_metrics.py` uses standard Python arithmetic, independent of Torch/loss code. It computes each eligible Euclidean landmark distance in supplied common units, averages eligible landmarks per image, then averages eligible images equally. In training verification, predictions are first inverted to source pixel units. This is **image-macro**, not patient-/video-macro averaging. Empty supervision gives undefined error plus zero coverage, never a manufactured zero error. Eligible nonfinite predictions/targets fail.

The implementation accepts all 68 source slots (indices 0–67), or an explicit valid index subset. Point and image coverage, eligible counts and undefined-normalizer counts accompany errors. The mask depends on target/geometry evidence, not favorable predictions.

Optional normalized error divides the per-image mean error by an explicitly supplied positive finite denominator before image averaging. Missing, zero, negative or nonfinite denominators leave normalized error undefined for that image and report support loss; no bounding-box/eye fallback is inserted. `interpoint_normalizers` supports a caller-declared pair only when both target points are available and separated. No indices are presumed to be eyes and no canonical InfAnFace interocular formula is asserted. The synthetic run therefore reports **no anatomical NME**. Hand calculations check 3–4–5 distances, unequal point-count image aggregation, coverage, subsets and undefined denominators.

## Environment and exact commands

Created `.venv-baseline` using available CPython 3.13.3, without modifying system Python or installing into the original `.venv`. The new complete hash lock pins 19 selected native/test wheels; every wheel digest matched PyPI release metadata before offline hash-enforced installation. Torch 2.14.1 is the new direct model dependency; Pillow 13.0.0 and all original development/build pins are reused. No torchvision, NumPy or mandatory CUDA dependency was installed. See [the tested environment record](../environments/landmark-baseline.md) and [complete lock](../environments/macos-arm64-py313-baseline.lock).

From the repository root, using the existing environments:

```sh
.venv-baseline/bin/python -m pytest -q
.venv-baseline/bin/python -m ruff check .
.venv-baseline/bin/python -m ruff format --check .
.venv-baseline/bin/python -m pip --disable-pip-version-check --no-cache-dir check
.venv-baseline/bin/infant-movement audit tests/fixtures/synthetic_metadata.json
.venv/bin/python -m pytest -q
.venv/bin/python -m pip --disable-pip-version-check --no-cache-dir check
```

Actual results: **248 tests pass with no skips** in the baseline environment; **224 pass with one optional Torch module skipped** in the preserved environment; lint and format (33 Python files) pass; both dependency checks pass; lock pins match installed distributions. Baseline tests include source inversion, target generation, masks/gradients, all-missing batches/epochs, invalid inputs, manufactured archive integration, validation/checkpoint metadata, CPU config bounds and prediction/RNG restoration. Synthetic tests require no infant images or downloads.

To repeat the fixed bounded experiment, select a **new unused run name**:

```sh
.venv-baseline/bin/python scripts/verify_synthetic_baseline.py --config configs/experiments/synthetic_landmark_baseline.json --run-dir runs/synthetic-landmark-baseline-rerun
```

The command returns 0 when its predeclared learning/reload criteria pass and 1 when they do not; invalid invocation or runtime failures also fail explicitly. It never overwrites an existing run. Fresh environment reproduction commands are in the environment record; installation is unnecessary for the existing `.venv-baseline`.

Early import-order lint issues were corrected. The initial sandboxed PyPI query failed DNS resolution; the authorized package query/download succeeded with network permission. The remaining first-import warning concerns Torch's unused optional NumPy bridge; the required pipeline and dependency checks pass without adding NumPy. No silent accelerator fallback or unresolved synthetic test failure remains.

## Preservation and next real-data pilot prerequisites

Raw files, original private inventory/evidence, historical reports/environment records/locks and research plan retain their pre-milestone hashes. The preprocessing modules are unchanged. `.venv` retains its interpreter/config and installed pins; `.venv-baseline`, private overlays/receipts and `runs/` are ignored. The core package remains free of mandatory runtime dependencies, with optional model imports confined to baseline modules. No commit, push, PR, remote publication, contact, new dataset/weight acquisition or background job occurred.

Before the first real InfAnFace pilot, decide and document:

1. Evidence-supported source indexing/frame conventions and independent private alignment adjudication; whether any remaining hypothesis is acceptable for a clearly limited pilot.
2. Explicit invalid-EXIF and out-of-bounds/missingness policy, retaining originals and coverage reporting.
3. Face crop source, label provenance and image resolution/resampling. Ground-truth crops require a ground-truth-crop label; no detector-backed result can be claimed without detection coverage. Current 32×32/full-image synthetic processing is not a validated real face protocol.
4. A defensible validation/grouping protocol despite unresolved subjects/videos; no random image split or one-file-per-subject invention. Official test membership/images remain reserved for later evaluation.
5. Canonical landmark/eye indices, exact NME denominator and aggregation/coverage definitions, plus separation of image-level feasibility from patient/video-independent claims.
6. Separate authorization for a bounded real-data training pilot. Any later pretrained HRNet path additionally needs code/checkpoint terms, training provenance, hashes and preprocessing parity review; no pretrained asset was acquired here.

Stopped after synthetic baseline implementation and verification. This remains exploratory geometry research, not a diagnostic or seizure-detection system.
