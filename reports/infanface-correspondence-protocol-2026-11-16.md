# Frozen-checkpoint image-correspondence protocol — 16 November 2026

Written before real-data reconstruction, inference or comparison calculation. Protocol
`infanface-image-correspondence-v1`, starting clean commit
`6aaea5fa0e29299ea1c7471e4925a4a9d9e36155`, branch `research/infanface-e0-audit`.
The user's present request authorizes this single evaluation-only comparison. No
training, new assets, alternative permutations, additional controls or test access.

1. **Frozen evidence.** Use only completed `runs/infanface-train-pilot-2026-10-28/`
   and completed `runs/infanface-mean-control-2026-10-28-verified/`. Verify every
   available historical pilot digest (49), mean-control output-integrity entry (11),
   historical code/config/protocol/lock and cross-record references. Preserve both
   runs, the failed mean-control preflight, all prior reports/verification records,
   original plan, raw data and environments unchanged. Snapshot historical files
   before edits, distinguishing newly frozen hashes from prior evidence. Verify
   receipt/archive and available original annotation-container digest without
   interpreting test labels or decoding test images. Any mismatch fails closed.
2. **Exact cohort and tensors.** Use the original ordered 32 usable selection rows,
   with no ranking/reselection/replacements. Confirm training membership before
   pixels, source rows, original annotation hashes, recorded boxes, raster sizes,
   EXIF tags, individual geometric mask counts and 68/68 valid slots per image.
   Reuse existing reconstruction/preprocessing contracts: source base-0 centres
   plus 0.5 to working edges, stored/stored annotation/EXIF, recorded 20%-expanded
   oracle boxes, uniform BICUBIC resize/black padding to 32×32, RGB CHW float32
   [0,1], no augmentation/normalization changes. Verify targets/masks/inverses and
   source targets exactly against mean-control private records before inference.
   Forward/inverse transforms stay with the destination. Source conventions and
   anatomical alignment remain **UNVERIFIED**; geometric masks are not visibility.
3. **Single image-only reassignment.** Destination position i gets the already
   prepared image tensor at `(i + 1) % 32` in the frozen usable order. Construct
   this mapping globally before batching; verify exact formula, bijection and
   zero fixed positions. Only `image` changes. Retain every destination target,
   mask, forward/inverse, crop and metadata field. No pixel rotation, landmark
   reindexing, recropping or source-target/transform transfer. Store mapping,
   ordered references and tensor digests privately. Check unchanged inputs after
   evaluation; refuse other sizes, mappings, membership or destinations.
4. **Environment and model.** Reuse unchanged `.venv-baseline` and its 19 exact
   dependency pins, CPU float32, one thread, deterministic algorithms, batch 8,
   original order, zero workers. All inventory text must match the pilot except
   the same editable repository/`infant_movement` egg's Git revision field; check
   original source/lock hashes strictly and record both environment digests. Load
   only verified epoch-20 fixed-final `final.pt` (historical SHA-256 retained
   privately). Evaluate in eval/no-grad mode, disable parameter gradients; no
   optimizer creation, backward, update, checkpoint save or epoch selection.
5. **Paired reproduction gate first.** Reproduce the original evaluator and
   independent float64 aggregate metrics against the pilot; independent metrics
   must also reproduce completed mean-control CNN records. Existing criterion:
   `abs(new-old) <= atol + 1e-6*abs(old)`, MSE atol `1e-7`, input/source pixel
   errors atol `1e-6`. Counts, masks, coverage and saved target/transform values
   must match exactly. Paired predictions must match the saved mean-control CNN
   float32 predictions exactly (prediction atol=rtol=0). Retain discrepancy
   details and stop before reassigned inference on any failure; never widen a
   tolerance, substitute evidence or rerun automatically.
6. **Exactly one reassigned evaluation.** After reproduction passes, evaluate the
   one global mapping with the same frozen model. Require finite predictions on
   eligible points. Report point-weighted normalized x/y-coordinate MSE (4,352
   scalars), image-macro Euclidean error in 32×32 input-edge pixels, and
   image-macro source-coordinate error using each destination's original inverse
   and source annotations, with no new half-pixel adjustment/clamping. Require
   matching 32 images and 2,176 points; no prediction-dependent attrition. Keep
   anatomical NME undefined. Reuse verified constant-control aggregates only as
   historical context; never fit its mean or calculate another control.
7. **Paired descriptive arithmetic.** Signed difference is reassigned minus paired;
   positive means increased error. For each metric report aggregate absolute and
   relative increase `(reassigned-paired)/paired`, and per-image mean, median,
   minimum, maximum, paired-lower/reassigned-lower/tied counts (absolute tie atol
   `1e-12`). Missing matched observations produce null/undefined metrics; unequal
   denominators or asymmetric missingness fail. Zero relative denominator,
   including 0/0, is null, with undefined counts recorded. No p-values, patient
   confidence intervals or inferred independent subjects.
8. **Software gate and records.** Before any real-image evaluation, pass synthetic
   tests of global mapping/batching, bijection, destination invariance, inversion,
   missingness/denominators/arithmetic, prohibited training operations and changed
   inputs; full relevant suites, Ruff lint/format, exact baseline inventory and
   both pip checks. Bind the gate to all current source/tests/scripts/config and
   protocol hashes. New ignored `runs/infanface-correspondence-2026-11-16/` only:
   resolved config, frozen reference/preservation hashes, cohort/permutation,
   environment/code/Git/checkpoint evidence, tensor hashes, reproduction details,
   private predictions/per-image differences, aggregate summary and status, then
   output-integrity manifest. Reject overwrite, historical paths/ancestors and
   symlink escapes. Retain failure status/evidence; no automatic repeat. Write
   dated aggregate report and README after completion, verify historical
   preservation/privacy, and stop.

Increased error supports a contribution from correct image–target correspondence
to this particular training fit. Little change or improvement weakens that evidence.
A single fixed reassignment establishes neither generalization, clinical utility
nor absence of shortcuts. Ground-truth-crop leakage, unknown source conventions,
subject/video independence and media-rights chain remain unresolved. The next
bounded step will be proposed from actual results, with separate authorization
required before any further experiment.
