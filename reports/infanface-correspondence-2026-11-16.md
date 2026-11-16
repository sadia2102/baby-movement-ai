# InfAnFace frozen-checkpoint image correspondence — 16 November 2026

## Result and permitted interpretation

**Reassigning prepared images increased error in all three aggregate metrics.** On
the exact original 32 official training examples, input-pixel image-macro error
increased **4.5561 → 5.3606 pixels**, a **0.8045 pixel / 17.6570% increase**. Correct
pairing had lower Euclidean error on 24 images, reassignment on eight, with no ties.
This supports a contribution from correct image–target correspondence to this
particular training fit. The eight improved images also show that correct pairing
was not uniformly better.

This was one evaluation-only comparison using the original fixed-final CNN, with
zero training updates, no checkpoint writes and exactly one reassigned condition.
Both conditions evaluate the same training images and annotation-assisted crops.
A single fixed reassignment cannot establish generalization, clinical usefulness
or absence of shortcut learning. Ground-truth-crop leakage remains. Source origin,
annotation frame, anatomical adjudication, subject/video independence and the
underlying-media rights chain remain **UNVERIFIED**. No seizure, clinical,
held-out, benchmark or patient-independent accuracy claim follows.

## Protocol and verified state

Read AGENTS, README, original research plan, E0 acquisition report, pilot and
mean-control protocols/reports, environment records, private pilot configuration,
selection/coverage/preparation/provenance/review/checkpoint/completion records,
and completed mean-control records/integrity manifest before implementation.
Starting branch was `research/infanface-e0-audit`, clean at
`6aaea5fa0e29299ea1c7471e4925a4a9d9e36155`, matching its locally recorded upstream.
No remote refresh, network acquisition, commit, push or upload occurred.

The [dated protocol](infanface-correspondence-protocol-2026-11-16.md) and fixed
`configs/experiments/infanface_correspondence.json` were written before real-data
reconstruction or inference. No post-result change to protocol, code, mapping,
metrics, tolerances or software gate was made. A newly written preservation
inventory froze **184 pre-existing files**, including historical runs, failed
mean-control preflight, private checks, raw data, reports, plan, code and locks.
All 184 matched immediately after calculation, before README/AGENTS updates;
the other 182 remain unchanged after those two permitted document updates.
New snapshots are preservation evidence, not retrospective authentication.

Verified the **49 historically recorded pilot digests**, all **15 pilot run files**,
all **11 mean-control output-manifest entries** and its manifest, and mean-control
historical runtime/source/config/protocol/lock references. Confirmed the completed
mean-control snapshot identifies exactly the same pilot files, ordered cohort and
checkpoint. Receipt/archive checks and the original annotation-container digest
match the earlier acquisition inventory; every selected row's annotation hash,
source row, explicit box, size, orientation and separate mask counts match. The
archive checksum reads compressed container bytes; the shared CSV is hashed as
bytes and test rows are discarded before numeric label interpretation. No
official test image member was opened.

The original usable order was joined directly, with no ranking, replacement,
additional examples or inferred identities. Applied recorded boxes through the
existing RGB float32 [0,1], BICUBIC uniform affine resize/black padding to 32×32.
Retained the unverified base-0-centre (+0.5 to working edges), stored/stored
annotation/EXIF hypotheses. Reconstructed targets, validity masks, inverses and
source targets match the saved mean-control evidence exactly. The original
pilot's all-valid mask counts uniquely determine every mask bit for this cohort.
Forward/inverse and all prepared tensor digests are saved privately and verified
unchanged after evaluation. No new anatomical alignment adjudication is claimed.

## Condition, reproduction and coverage

Destination position i received the already prepared image tensor at
`(i + 1) % 32`, globally across the frozen usable order before batch handling.
The map is bijective, with zero fixed positions. Only images changed; targets,
geometric masks, crop/metadata and both transforms remained with the destination.
No raster rotation, landmark reindexing, crop/padding change or source-target
transfer occurred. The permutation and references remain private.

Loaded only the digest-verified original epoch-20 fixed-final state on CPU, using
the original batch size eight, one thread, deterministic algorithms, zero workers,
eval/no-grad and disabled parameter gradients. No optimizer was constructed or
restored. The model's state digest remained unchanged. No backward, optimizer
update, epoch/checkpoint selection, retraining or replacement checkpoint occurred.

Before reassigned inference, reproduced the original evaluator and independent
float64 accumulation against the pilot and completed mean-control CNN metrics.
Predeclared tolerances were unchanged: `abs(new-old) <= atol + 1e-6*abs(old)`,
MSE atol `1e-7`, pixel-error atol `1e-6`; exact counts/coverage/target evidence and
exact saved paired predictions (prediction atol=rtol=0). Original-evaluator
metrics matched the pilot **exactly**. Independent MSE differs from the pilot's
float32 accumulation by **2.893275063975409e-10**, within tolerance, and equals
the prior independent result. Input/source errors match exactly. Every saved
paired prediction matches exactly; all discrepancy lists are empty. No
tolerance was widened or result substituted.

| Coverage / denominator | Both conditions |
|---|---:|
| Exact original usable / selected images | 32 / 32 |
| Eligible / selected landmark slots | 2,176 / 2,176 |
| Coordinate scalars for MSE | 4,352 |
| Geometrically eligible slots per image | 68 |
| Newly excluded images / points | 0 / 0 |
| Image / point coverage | 100% / 100% |
| Undefined anatomical-NME denominator images | 32 |
| Test image members opened / training updates | 0 / 0 |

These masks measure geometric eligibility, not visibility or verified anatomical
accuracy. No validation/project split or patient weighting was invented.

## Aggregate and descriptive paired results

MSE averages all eligible normalized x/y-coordinate scalars in float64.
Euclidean errors average landmarks within image and images equally. Source error
uses each destination's original inverse and source annotations, without new
half-pixel offsets or clamping; varying crop/source resolution affects its units.
All 68 array slots enter, without an invented anatomical NME denominator.

| Training-cohort metric | Correctly paired | Reassigned | Reassigned − paired | Relative increase |
|---|---:|---:|---:|---:|
| Normalized-coordinate MSE | 0.01366299210178018 | 0.01853915360711847 | 0.004876161505338289 | 35.6888% |
| Image-macro input-pixel error | 4.556128670173429 | 5.360606281878322 | 0.8044776117048933 | 17.6570% |
| Image-macro source-pixel error | 82.98714856711695 | 96.23744434766502 | 13.250295780548072 | 15.9667% |

Relative increase is `(reassigned-paired)/paired`. Undefined observations produce
null metrics; a zero relative denominator, including 0/0, produces null rather
than infinity or fabricated 0%. All aggregate/per-image denominators were
positive here, with zero undefined relative comparisons. Anatomical NME remains
undefined separately. Conditions share every eligible point; no attrition hides
nonfinite predictions.

Signed per-image differences below are reassigned minus correctly paired,
positive favoring correct pairing. Each row uses 32 matched images, 68 points /
136 scalars per image. Absolute tie tolerance `1e-12`; no patient-independent
sampling, p-value or confidence interval is asserted.

| Metric | Mean difference | Median | Minimum | Maximum | Paired lower / reassigned lower / ties |
|---|---:|---:|---:|---:|---:|
| Normalized MSE | 0.0048761615053383 | 0.0042136137011897 | -0.0032658979137834 | 0.0161423075780976 | 23 / 9 / 0 |
| Input-pixel error | 0.8044776117048925 | 0.7233229197420599 | -0.8700819659288630 | 3.6764892954383206 | 24 / 8 / 0 |
| Source-pixel error | 13.250295780548068 | 9.316116681918057 | -33.77089062179073 | 81.22743527636459 | 24 / 8 / 0 |

For context only, the verified historical constant control recorded normalized
MSE **0.015497068270144174**, input error **4.826836362962258** and source error
**87.2972359838528**. These values were read from integrity-verified records;
the mean was not refitted or reevaluated, and no additional control was run.

## Software verification and errors

Reused unchanged `.venv-baseline`: CPython 3.13.3, Torch 2.14.1, Pillow 13.0.0,
macOS 27.0.1 arm64; pytest 8.4.2 and Ruff 0.14.2. No dependencies, environment
records or locks changed. All **19 exact pinned distributions plus the editable
project** passed the original lock/inventory check. The current `pip freeze`
differs from the pilot only in the documented same editable repository/egg Git
revision field. All dependency text and historical source/lock hashes were
checked strictly; no relaxation was applied to them. Both raw environment
digests and the permitted distinction are retained privately. Installed versions
were checked; no fresh wheel download or reinstallation is claimed.

Before real-image evaluation, the final software gate passed:

| Check | Actual result |
|---|---|
| Full `.venv-baseline` synthetic suite | **338 passed** |
| Ruff lint / format | Passed / **50 files already formatted** |
| Exact baseline installed inventory | **19 pins + editable project**, matched |
| Baseline and preserved `.venv` pip checks | Both passed |
| Preserved `.venv` suite | **250 passed, five expected Torch-module skips** |
| Synthetic metadata/group audit | Exit 0; unknown-identity limitation retained |

The 28 added tests use manufactured pixels only. They exercise global wraparound
and cross-batch mapping (batch sizes 1/7/8/32), exact authorized bijection,
unchanged destination targets/NaNs/masks/crops/forward/inverse/metadata, missing
observations, zero denominators, destination inverse arithmetic, paired metric
sign/ties/denominators, eval/no-grad/frozen parameters, no optimizer/backward/
checkpoint save/mean refit, changed cohort/order/annotation/box/membership/
checkpoint/config/mean evidence/software rejection, reproduction discrepancy
retention, historical/path/symlink protection and second-comparison refusal.
Existing coordinate round-trip and preprocessing tests also ran in the full suite.

During development, the first targeted test run had **26 passes and two failures**:
the synthetic preservation fixture included a check file it subsequently rewrote.
The fixture was corrected to exclude that mutable synthetic gate; no production
hash check was loosened. An import-shadow lint finding and long lines were also
corrected before the final passing gate. The targeted suite then passed 28/28.
**There was no real preflight or reproduction failure in this phase.** The
historical failed mean-control preflight remains unchanged. The documented
optional NumPy-bridge warning still appears; this path uses no NumPy bridge.

## Exact commands, changed files and private records

Executed final gate and the one real comparison from the project root:

```sh
.venv-baseline/bin/python scripts/verify_infanface_correspondence_checks.py --output reports/private/infanface-correspondence-software-verification-2026-11-16.json
.venv-baseline/bin/python scripts/run_infanface_correspondence.py --config configs/experiments/infanface_correspondence.json --run-dir runs/infanface-correspondence-2026-11-16 --checks reports/private/infanface-correspondence-software-verification-2026-11-16.json
```

The verifier invokes the unchanged `scripts/verify_infanface_pilot_checks.py`,
retaining its base record and all exact subprocess commands/stdout/stderr, then
binds the result to current code/tests/scripts and this protocol/config/preservation
inventory. Targeted development verification also ran:

```sh
.venv-baseline/bin/python -m pytest -q -p no:cacheprovider tests/test_image_correspondence.py
.venv-baseline/bin/python -m ruff format --no-cache --check .
```

These are historical executed commands, not authorization to repeat the
experiment. Existing destinations, historical descendants/ancestors, symlink
escapes and another attempted run under this protocol are rejected. Failed
records must remain intact; changing/deleting them to bypass refusal is not
authorized.

New source: `engine/image_correspondence.py`, `evaluation/image_correspondence.py`;
new scripts: `run_infanface_correspondence.py`,
`verify_infanface_correspondence_checks.py`; new synthetic tests:
`test_image_correspondence.py`; fixed JSON config, protocol and this report.
README and AGENTS record completion. Existing code/tests, original research plan,
historical reports, raw data, locks, environments and historical runs are unchanged.

The new ignored run holds resolved config, private frozen pilot/mean references,
cohort/permutation and per-tensor hashes, code/environment/Git/checkpoint evidence,
paired reproduction checks, paired/reassigned predictions, destination targets
and transforms, per-image differences, aggregate metrics, completion status and
an output SHA-256 manifest. It contains no imagery or replacement checkpoint.
Selected references, maps, predictions and sample evidence remain Git ignored.
Private software/preservation records are also ignored. Verification of the new
manifest and historical preservation required no further model execution.

## Next bounded step — proposed, not executed

Propose one **metadata-only provenance evidence audit of this exact cohort**:
review already retained source/group evidence and list documented versus missing
relationships without inferring identities, opening test images, acquiring data,
running a model or creating splits. This is justified because correspondence
now contributes to the observed fit, while unknown grouping still prevents a
defensible independent validation claim. Stronger coordinate/anatomical evidence
and a defensible validation protocol remain prerequisites to later generalization
work. This proposal needs separate authorization; no further experiment was run.
