# Synthetic heatmap adapter — 16 November 2026

## Delivered result

Implemented an optional-Torch adapter for Gaussian heatmap targets, deterministic
argmax decoding and masked heatmap MSE. Initial input/heatmap sizes are **32×32 /
8×8**, sigma **1.0 heatmap pixels**, with **68 slots** supported. Sizes are always
`(width,height)`; tensor axes are `[B,K,Hh,Wh]`. Dimensions are configurable and
non-square, unequal, noninteger axis scales are tested. This is adapter engineering,
not a new model or experiment.

**61 new synthetic tests pass; the full baseline suite passes 468 tests.** The
preserved environment passes 319 tests with six expected Torch-module skips.
The saved offline verifier passes **12/12** independent arithmetic checks. No
HRNet, CNN replacement, optimizer or training loop was added. No real infant
image/annotation, historical checkpoint/prediction or official test sample was
opened. No historical metric was recalculated, split/identity created, dependency
installed, network accessed, contact made, upload/commit/push or background job run.

The [dated protocol](heatmap-adapter-protocol-2026-11-16.md) preceded coding.
Starting branch was `research/infanface-e0-audit`, HEAD
`6aaea5fa0e29299ea1c7471e4925a4a9d9e36155`, with coordinate/conventions,
correspondence and provenance work already uncommitted. Read AGENTS/README/plan,
latest coordinate/metric plan/report and source registry, existing geometry,
preprocessing/tensor/metric interfaces, relevant tests and environment records.
No source website was refreshed. Existing implementation and scientific assumptions
are preserved: source origin/frame, independently confirmed local anatomical order,
subject/video independence and parts of the rights evidence remain **UNVERIFIED**.

## Target and validity contract

`datasets/landmark_heatmaps.py` is imported only explicitly. Core imports remain
dependency-free; Torch stays optional in unchanged `pyproject.toml`. This adapter
currently accepts **CPU float32 or float64** tensors; no MPS/CUDA parity is claimed.

`generate_heatmaps(input_edges, eligibility, config=...)` accepts nonempty `[B,K,2]`
coordinates and matching CPU bool `[B,K]` masks. Inputs must already be in the
working **input-edge frame**. Raw source numbers need the caller's declared full
origin/frame/crop/resize/padding transform first. Existing normalized tensor
targets need multiplication by `(W,H)` to obtain edges; no implicit conversion or
assumption about source coordinates occurs.

For input `(W,H)`, heatmap `(Wh,Hh)`, fractional centre is
`(hx,hy)=(x*Wh/W,y*Hh/H)`. Cell `(r,c)` is centred at `(c+0.5,r+0.5)` and stores

```text
exp(-0.5 * (((c+0.5-hx)/sigma)^2 + ((r+0.5-hy)/sigma)^2))
```

Every finite-grid cell is sampled; there is no finite-radius cutoff or peak/sum
normalization. Out-of-grid Gaussian tails are unrepresented. Border centres are
not relocated. For example, input edge `(0,0)` is eligible and yields peak
`exp(-0.25)` at the first cell for sigma 1; normalizing that peak to one would
change the declared target. Input `(10,14)` maps to heatmap centre `(2.5,3.5)`,
with unit peak at row 3/column 2, axial neighbour `exp(-0.5)` and diagonal
neighbour `exp(-1)`.

Results retain detached copies of original input coordinates, float64 heatmap
centres, caller masks, present/footprint masks, effective eligibility and coverage.
No input is mutated. Effective eligibility is caller eligibility AND the half-open
footprint `0<=x<W, 0<=y<H`. Finite out-of-bounds points stay numerical and receive
false effective masks; they are neither clamped nor moved into the grid.

Missing tensor pairs must be NaN/NaN with false caller eligibility. The caller's
original None metadata remains untouched; there is no automatic None conversion.
Partial-NaN pairs, infinity, or eligible missing coordinates fail. Invalid slots
store zero maps **with false masks only**; they are never supervised negative
examples. Eligible Gaussians that lose all representable support fail explicitly.
These geometric/caller masks do not infer visibility, occlusion or anatomical
correctness. Target generation detaches coordinate gradients.

## Decoder, inverse and quantization

`decode_argmax` requires maps `[B,K,Hh,Wh]`, bool eligibility, a complete recorded
float64 `[B,3,3]` nonsingular affine inverse, explicit integer source index base
0/1 and annotation frame stored/display. These are caller declarations, not
verification of dataset conventions. The inverse already includes all origin,
orientation, crop and padding operations.

The decoder returns heatmap-edge cell centres, input edges and source annotation
numbers, preserving slot order. Input edges use independent factors `W/Wh` and
`H/Hh`. Source conversion reuses `torch_landmarks.decode_coordinates` and the
existing normalized-edge/inverse contract. **No extra half pixel** is added and
the legacy integer-truncated transform is not adopted. Invalid/projective/singular
or nonfinite inverse matrices and eligible source-arithmetic overflow fail.

The first exact maximum in row-major order wins. Peak multiplicity and ambiguity
are exposed. A nonflat tied map remains supported with ambiguity flagged. Flat
maps of any value, all-zero maps and single-cell maps are unsupported: decoded
mask false and coordinate pairs NaN, never a valid top-left point. Missing-mask
maps are not inspected; any nonfinite value anywhere in an eligible map fails
instead of shrinking coverage. Requested and supported coverage are separate.
Nonflat negative scores are allowed; raw peaks can be negative or exceed one and
are **not calibrated confidence probabilities**. No threshold was fitted.

Argmax is nondifferentiable and quantized. An ideal sampled Gaussian selects a
nearest cell centre, at most half a heatmap cell per axis. At the initial sizes
that is **two input pixels per axis**, Euclidean bound `2*sqrt(2)≈2.8284` input
pixels. This is a representation bound for an ideal Gaussian, not a bound on a
model's error. Float arithmetic/ties are reported explicitly. Continuous affine
inversion cannot restore the subpixel information lost by argmax.

For manufactured input `(11.2,14.8)`, the centre is `(2.8,3.7)`, raw peak
`exp(-0.065)=0.9370674633774034`, and decoded input edge `(10,14)`. The displacement
is `(-1.2,-0.8)`, norm **1.4422205101855956 input pixels**. This expected
quantization is not an affine round-trip failure or subpixel recovery claim.

## Masked loss

`masked_heatmap_mse(prediction,target,eligibility)` selects eligible slots before
subtraction/squaring, casts selected values to float64 and divides their squared
error sum by **eligible_slots × Hh × Wh**. This weights eligible landmark maps/
cells equally across the batch, not images equally. Coverage reports images,
slots, cells and denominator. Eligible nonfinite inputs or nonfinite loss fail.

Entirely absent supervision returns `HeatmapLoss(loss=None, denominator=0, ...)`,
never a fabricated zero loss. Invalid NaN maps have zero gradient contribution.
Autograd through predictions is retained; no backward or optimizer is called by
the utility. The verifier/tests call backward only on manufactured leaf tensors.

Independent hand case: valid predictions `[1,3]`, `[2,4]`, `[0,2]` against zero,
plus one masked NaN map, give squared sum **34**, six cells and loss **34/6**.
Valid gradients are `2*prediction/6`; invalid gradients are zero. Unequal per-image
support is explicit. Existing image-macro landmark metrics and historical loss/
trainer behavior remain unchanged.

## Working synthetic example

Run with the existing baseline interpreter:

```python
import torch
from infant_movement.datasets.coordinate_conventions import source_centres_to_edges
from infant_movement.datasets.landmark_heatmaps import (
    HeatmapConfig, generate_heatmaps, decode_argmax, masked_heatmap_mse,
)

config = HeatmapConfig(input_size=(32, 32), heatmap_size=(8, 8), sigma=1.0)
edges = torch.tensor([[[10.0, 14.0]] * 68])
target = generate_heatmaps(edges, torch.ones((1, 68), dtype=torch.bool), config=config)
# Manufactured origin-only inverse. A real caller must supply its COMPLETE inverse.
inverse = torch.tensor([source_centres_to_edges(source_index_base=0).inverse().matrix],
                       dtype=torch.float64)
decoded = decode_argmax(target.heatmaps, target.eligible, inverse=inverse,
                        source_index_base=0, annotation_frame="stored", config=config)
assert decoded.input_edges[0, 0].tolist() == [10.0, 14.0]
assert decoded.source_coordinates[0, 0].tolist() == [9.5, 13.5]
loss = masked_heatmap_mse(target.heatmaps.clone().requires_grad_(),
                          target.heatmaps, target.eligible)
assert loss.denominator == 68 * 8 * 8 and loss.loss.item() == 0.0
```

This example is manufactured arithmetic. It supplies no dataset convention,
anatomical guarantee, detector/crop accuracy, NME result or real-model pilot.

## Actual checks, commands and failures

Reused unchanged CPython **3.13.3**, macOS **27.0.1 arm64**, Torch **2.14.1**,
Pillow **13.0.0**, pytest **8.4.2**, Ruff **0.14.2**. Exact installed inventories
match the retained **19 baseline pins** and **ten preserved-environment pins**,
each plus editable project. No dependency, lock or environment changed. The
known first-import optional NumPy-bridge warning remains; this path uses no bridge.

| Verification | Actual result |
|---|---|
| Targeted new suite | **61 passed** |
| Full `.venv-baseline` synthetic suite | **468 passed**, no skips |
| Full preserved `.venv` suite | **319 passed, six expected Torch-module skips** |
| Ruff lint / formatting | Passed / **60 files already formatted** |
| Baseline and preserved dependency checks | Both passed |
| Exact installed inventories | 19 and ten pins + project, matched |
| Synthetic metadata/group CLI | Exit 0; unknown-identity warning retained |
| Saved standalone arithmetic verifier | **12/12 passed** |

The new tests cover independent Gaussian/cell/half-pixel values, both float dtypes,
68-slot ordering, non-square and unequal noninteger scale factors, fractional
centres, default argmax quantization, crop/padding and batch-specific affine
inversion/shear, origin declarations, OOB/missing/unequal masks, border truncation,
two/four-way ties, flat/single-cell/unsupported maps, nonfinite/config/matrix/shape
errors, all-missing loss, hand gradients, target/decoder detachment and lack of
mutation. Tests require no infant data, image, checkpoint or pretrained asset.
A subprocess audit hook denies network and research-artifact opens during the
offline verifier; asset arguments and public output destinations are rejected.
Existing output/path/symlink refusal uses the already tested private-run helper.
Core-only import is checked without importing Torch.

The first targeted run had **53 passes and one failure** because the test used
nested `pytest.approx`, unsupported by the retained pytest version. Assertions
were corrected to compare coordinate pairs separately; no tolerance changed.
Ruff also caught default-constructor calls and long lines; the immutable default
config now has a named module-level instance and style findings were corrected.
The next targeted run passed 54/54, then seven additional verifier/immutability/
tie tests brought the final targeted suite to 61/61. No final software/verifier
failure occurred; no failed real-data operation exists because none was attempted.

Exact targeted and final executed commands:

```sh
.venv-baseline/bin/python -m pytest -q -p no:cacheprovider tests/test_landmark_heatmaps.py
.venv-baseline/bin/python -m ruff check --no-cache .
.venv-baseline/bin/python -m ruff format --no-cache --check .
.venv-baseline/bin/python scripts/verify_infanface_pilot_checks.py --output reports/private/heatmap-adapter-software-verification-2026-11-16.json
.venv-baseline/bin/python scripts/verify_heatmap_adapter.py --run-dir runs/synthetic-heatmap-adapter-2026-11-16
```

The existing software-gate script executed both full pytest suites, lint/format,
both `pip --disable-pip-version-check --no-cache-dir check` commands and the
synthetic metadata audit; exact absolute command arrays/stdout/stderr are retained
in its ignored record. Its name does not trigger pilot preparation/execution.
The full baseline suite includes only the already-existing manufactured-model/
temporary synthetic-checkpoint tests, expressly allowed as software verification.
No training-experiment entry point was invoked.

The standalone verifier has only optional `--run-dir`; there is no dataset,
archive, checkpoint, model or parameter-search argument. Without it, checks print
without saving. A saved run refuses an existing destination, historical ancestors/
descendants and resolved symlink escapes. It uses CPU/one thread/deterministic
algorithms and restores thread/determinism settings afterward. Fixed JSON config
is checked exactly, with no downloads or hidden initialization. Failure retains
status in the new directory; no historical run is writable through this command.

Saved arithmetic results include:

| Manufactured quantity | Actual value |
|---|---:|
| Float32 Gaussian axial neighbour | 0.6065306663513184 |
| Fractional Gaussian raw peak | 0.9370674633774034 |
| Fractional argmax input error | 1.4422205101855956 pixels |
| Non-square `(30,18)→(7,5)` crop/padding decoded source | `(9.5,4.5)` |
| Non-square source max absolute error | 0 source pixels |
| Masked MSE / cell denominator | 5.666666666666667 / 6 |
| Maximum hand-gradient discrepancy | 0 |

The 68-slot verification fixture deliberately contains one missing masked pair
and one numerical OOB pair: **67 requested eligible / 66 effective eligible**,
4,224 supervised heatmap cells, with exclusions explicit. These are manufactured
coverage counts, not InfAnFace sample coverage. Tolerances remained geometry
`1e-10`, float64 arithmetic `atol=1e-12,rtol=0`, float32 Gaussian
`atol=1e-7,rtol=1e-6`, with exact discrete masks/counts/tie order.

## Records, changed files and preservation

New files: `datasets/landmark_heatmaps.py`, `tests/test_landmark_heatmaps.py`,
`scripts/verify_heatmap_adapter.py`, `configs/experiments/synthetic_heatmap_adapter.json`,
the protocol and this report. README/AGENTS record completion. Existing model,
trainer, preprocessor, tensor adapter, coordinate/metric utilities, source registry,
tests, research plan and historical reports are unchanged.

The new ignored `runs/synthetic-heatmap-adapter-2026-11-16/` holds resolved config,
actual arithmetic, exact environment text/inventory, code/config/protocol/lock and
environment hashes, Git state, completion status and output SHA-256 manifest.
Private preservation and full software-check records remain under ignored
`reports/private`. Generated caches/run records are not tracked. No image, model,
checkpoint or dataset reference is saved by the new verifier.

Before edits, snapshotted **123 public files** and **141 sealed artifact** size/
mtime/inode records. All matched immediately after synthetic verification, before
README/AGENTS updates; the other **121 public hashes** remain unchanged afterward.
Sealed artifacts were never reopened for content hashing, so their statistical
preservation is not a fresh content-authentication claim. Earlier uncommitted
coordinate/convention, correspondence and provenance additions remain intact.

## Next bounded proposal and real-pilot prerequisites

Propose a **synthetic-only small heatmap-head forward/loss/backward interface
smoke check**, under a new protocol and authorization, with manufactured tensors
and no optimizer/training experiment or real-data access. This would check the
adapter/model tensor contract before a separate model effort; it was not executed.

A real heatmap-model pilot still requires a separately authorized model/protocol,
stronger source-origin/frame and independently adjudicated local anatomical/target
semantics, explicit crop/visibility/occlusion policies, justified metric/coverage
rules, access/derivative-rights review and a defensible evaluation scope. Subject/
source grouping remains unresolved; no held-out generalization protocol has been
established. The adapter supplies software arithmetic, not those scientific/data
prerequisites. Stop after this implementation, verification and documentation.
