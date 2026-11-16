# Synthetic heatmap adapter — fixed implementation protocol

Written **16 November 2026 before coding**. Starting HEAD `6aaea5f`, branch
`research/infanface-e0-audit`, with coordinate/convention, correspondence and
provenance work already uncommitted. Preserve that work, historical code/reports/
runs/locks/raw assets and the original plan. Only README/AGENTS receive additive
status updates. No real-data processing, model experiment, new training run,
assets, installation, network, contact, upload, commit/push or background job.

1. **Interface/config:** a separately imported optional-Torch adapter. CPU float32
   or float64 coordinates `[B,K,2]`, explicit bool eligibility `[B,K]`, nonempty B/K,
   including K=68. Initial input `(W,H)=(32,32)`, heatmap `(Wh,Hh)=(8,8)`, sigma
   1.0 heatmap pixels. Sizes are `(width,height)`; map axes `[B,K,Hh,Wh]`.
   Dimensions/sigma are configurable; only manufactured non-square cases accompany
   this fixed initial setting, without a parameter sweep or model.
2. **Targets:** input coordinates are pixel **edges**. Heatmap centres are
   `(x*Wh/W,y*Hh/H)`, without rounding. At cell `(r,c)`, sample
   `exp(-((c+0.5-hx)^2+(r+0.5-hy)^2)/(2*sigma^2))` over the entire finite grid.
   No peak/sum normalization, finite-radius cutoff, clamping or relocated targets.
   Infinite Gaussian tails outside this grid are simply unrepresented, including
   at borders. Preserve original input tensors/coordinates in detached copies.
3. **Validity:** retain caller eligibility separately from effective eligibility.
   Effective = caller eligible AND half-open input footprint `0<=x<W, 0<=y<H`.
   Finite out-of-bounds numbers remain numerical with false effective masks.
   Missing tensor pairs are NaN/NaN and require false caller masks; None stays in
   the caller's original metadata (no automatic None-to-coordinate conversion).
   Partial-NaN pairs, infinities or nonfinite eligible coordinates fail. Invalid
   slots store zero maps only alongside false masks; they are never supervised
   negatives. Reject eligible Gaussians that lose all representable support.
4. **Decoder:** explicit `[B,K,Hh,Wh]` maps plus bool eligibility, complete recorded
   float64 `[B,3,3]` nonsingular affine inverses, declared source index base 0/1
   and stored/display frame. Finite eligible maps only. Argmax selects the first
   exact maximum in row-major order; expose peak multiplicity/ambiguity. Flat maps,
   including all-zero and single-cell maps, are unsupported: false decoded mask
   and NaN coordinates, never a valid top-left point. Nonflat scores may have any
   sign; raw peaks are uncalibrated scores. Return cell-centre heatmap edges,
   per-axis-scaled input edges, and source numbers through the existing normalized-
   edge inverse contract, adding no extra 0.5 and retaining landmark slot order.
5. **Quantization:** argmax returns cell centres, not exact subpixel centres.
   For an ideal sampled Gaussian, nearest-centre quantization is at most half a
   heatmap cell per axis (initially two input pixels per axis, Euclidean bound
   `2*sqrt(2)`). This is not a bound on arbitrary model error. Affine inversion
   remains continuous/reversible; it does not undo argmax quantization. No legacy
   integer transform, soft-argmax, refinement, NME switch or anatomical swap.
6. **Loss:** separate masked heatmap MSE selects eligible slots **before** any
   subtraction/squaring, so invalid NaNs cannot poison values/gradients. Float64
   accumulation, denominator `eligible_slots*Hh*Wh`, point/cell-weighted across B/K
   rather than image-macro. Report requested slots, eligible slots/images/cells,
   coverage and explicit no-loss `None` for denominator zero. Eligible nonfinite
   values or nonfinite loss fail. No optimizer/trainer/model is added.
7. **Checks/records:** independent hand values, peaks, unequal scales, fractional
   centres/quantization, border/ties/flat maps, source origins/crop/padding/inverse,
   missing/OOB/masks, bad inputs and masked-loss gradients. Geometry tolerance
   `1e-10`, float64 arithmetic `atol=1e-12,rtol=0`, float32 Gaussian
   `atol=1e-7,rtol=1e-6`; discrete masks/counts/tie order exact. Offline verifier
   has no dataset/archive/checkpoint argument or model operation. New ignored
   `runs/synthetic-heatmap-adapter-2026-11-16/` only: resolved config, arithmetic
   results, code/config/protocol/environment/Git hashes, checks and completion/
   failure status; refuse overwrite, historical nesting and symlink escapes.
   Run targeted tests, both full relevant synthetic suites, lint/format, both
   dependency checks and exact existing inventories. Existing manufactured-model
   tests are software verification only; do not launch an experiment entry point.

Update README/AGENTS and write a dated report with commands, actual results,
failures, assumptions and one next bounded proposal, then stop. Source origin,
frame, independently confirmed local anatomy and subject/video independence stay
**UNVERIFIED**. Nothing authorizes a real heatmap pilot or clinical claim.
