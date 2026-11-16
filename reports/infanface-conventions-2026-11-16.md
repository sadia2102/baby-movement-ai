# InfAnFace synthetic coordinate and metric compatibility — 16 November 2026

## Delivered result

Implemented and verified **dependency-free coordinate adapters and an opt-in source-referenced NME wrapper**, without recalculating any infant-data result. The new adapters preserve fractional source coordinates through explicit origin/crop/resize/padding/inverse transforms. NME uses the pinned official evaluator's target-distance pair, zero-based slots **36 and 45**, with explicitly assumed Multi-PIE68 order. Missing/zero denominators remain undefined, with coverage counts; fraction and percent have separate names.

The new legacy transform reference demonstrates an important boundary: integer-truncated upstream pixel arithmetic is not generally invertible, so it is not a substitute for the project's continuous reversible geometry. This is a narrow arithmetic comparison, not a claim of full HRNet preprocessing/heatmap/runtime parity. Source pixel origin, global annotation frame, independently confirmed local anatomical order and subject/video grouping remain **UNVERIFIED**. Real-data NME remains uncomputed and historical metrics remain unchanged.

The [bounded plan](infanface-conventions-protocol-2026-11-16.md) was written before implementation. Its introductory wording was clarified after verification to distinguish forbidden real-data model/checkpoint operations from the existing synthetic model/checkpoint tests explicitly included in the planned full-suite check. No experiment, image inspection, historical checkpoint/prediction loading, split creation or additional data/model-weight acquisition occurred.

## Pinned primary evidence

Source metadata and exact fetched UTF-8 SHA-256s are in `configs/datasets/infanface_conventions.json`. The official repository was inspected read-only and pinned to commit **`329abb24a858e3591d2b63f3bc0f3f5f41ac0b56`**. Four small documentation/code files were read, not imported, executed or vendored. A public documentation/code fetch is distinct from acquiring research data or weights; no dataset/image/CSV/weight endpoint was opened. The offline verifier does not refresh sources or use a network.

| Evidence | Supported statement | Remaining boundary |
|---|---|---|
| [Official evaluator](https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking/blob/329abb24a858e3591d2b63f3bc0f3f5f41ac0b56/lib/core/evaluation.py) | For 68 points, denominator is ground-truth Euclidean distance between zero-based slots 36/45; numerator averages distances across all 68 points | Observed implementation convention, not independent local slot/annotation adjudication |
| [Dataset reader](https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking/blob/329abb24a858e3591d2b63f3bc0f3f5f41ac0b56/lib/datasets/face300w.py) | Reader adds one before the pixel transform and subtracts one for heatmap target generation; it also contains a positive-y target guard | These implementation operations do not settle the raw archive's origin/frame; the positive-y guard is not adopted |
| [Transform source](https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking/blob/329abb24a858e3591d2b63f3bc0f3f5f41ac0b56/lib/utils/transforms.py) | One-based transform interface, centre/scale extent `200*scale`, integer truncation, optional inverse | Only zero-rotation, square-output arithmetic is reproduced; no crop/resize/rotation or full decoder equivalence asserted |
| [Paper v3, III and IV-B](https://arxiv.org/html/2110.08935v3) | Author-declared Multi-PIE layout, outer-eye-corner normalization, and self-occluded landmarks assigned to the nearest visible facial point | These targets do not automatically describe hidden anatomical projections; per-point visibility and exact local raw conventions remain unresolved |
| [Official README](https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking/blob/329abb24a858e3591d2b63f3bc0f3f5f41ac0b56/readme.md) | Dataset/collection-batch structure and stated noncommercial research conditions | No new supplied subject/video mapping or redistribution clearance established; the separate author-hosted README was inaccessible through the browser |

The prior documents remain valid historical records of missing evidence. This phase adds a documented **implementation reference** for metric indices and an author-stated occlusion convention; it does not promote local anatomy, frame/origin or grouping to verified status. No old report is rewritten to suggest this information was already validated there.

## Interfaces and assumptions

`datasets/coordinate_conventions.py` adds:

- `source_centres_to_edges(source_index_base=...)`: explicit integer 0 or 1 only. Base-0 centre `(0,0)` and base-1 centre `(1,1)` both become working edge `(0.5,0.5)`. Neither choice establishes the dataset's actual origin.
- `encode_normalized_edges(points, forward=..., output_size=(W,H))`: apply the caller's complete source-to-output-edge affine and normalize x/W, y/H.
- `decode_normalized_edges(points, inverse=..., output_size=(W,H))`: denormalize and apply the recorded inverse, with no extra half-pixel offset. Preserve `None` and numerical out-of-bounds values; reject malformed/nonfinite points. The affine must encode the caller's explicit frame, origin and crop/padding assumptions. No implicit orientation inference or semantic slot swaps.

`evaluation/infanface_conventions.py` adds `infanface_reference_nme(..., slot_order="assumed-multipie68", units="source-pixels" | "input-pixels")`. Exactly 68 slots and boolean masks are required. Coordinates must share the declared pixel frame/units. Independently normalized x/W,y/H values from a non-square image are not accepted as a common pixel metric. Callers must convert predictions and targets to compatible pixel coordinates first.

For every eligible image, the wrapper averages eligible landmark Euclidean distances and divides by the positive finite **target** slot-36/45 distance, then averages defined image NMEs equally. With all 68 points eligible it reproduces the upstream metric formula arithmetically. Partial masks are an explicitly labeled **project extension**; the reference evaluator has no equivalent mask contract. Both anchors must be eligible. Missing, zero or nonfinite denominators have no fallback; missing observations are not zero errors. Eligible nonfinite inputs and arithmetic/percent overflow fail explicitly.

The existing generic metric fields remain present. `nme_fraction` and `nme_percent` differ by exactly 100. Point/image and normalized-image coverage remain separate. `undefined_normalizer_images` retains the generic meaning (eligible images whose denominator is undefined), while `undefined_normalizer_images_total` also counts no-target images. `fully_eligible_reference_images`, `masked_extension_normalized_images`, `normalizer_defined_images` and `no_eligible_point_images` make denominators explicit. No patient weighting or clinical probability is implied.

`legacy_hrnet_pixel_reference` is separately named and requires explicit forward/inverse direction, square integer resolution, centre and positive finite scale. It subtracts one at the interface, applies the unrotated affine, truncates toward zero and adds one. For centre `(50,50)`, scale 1 and resolution 100, `(11,21)` maps to `(31,36)`. Fractional `(10.25,20.75)` maps to `(30,35)` and returns `(9,19)` through the quantized inverse. This loss is expected quantization, not a failure of the reversible adapter. A negative-value example distinguishes truncation from floor. The upstream positive-y guard is not copied: under our declared base-0 hypothesis, a y=0 boundary annotation can be eligible.

Existing reader, image preprocessor, tensor adapters, training loops, metric functions, prior test files and experiment entry points were **not modified**. No automatic switch to NME or change of historical fit results is introduced.

## Synthetic checks and actual verification

The 35 added tests require no infant data, images, model weights, NumPy, Pillow or Torch. They cover a hand-calculated 3–4–5 eye denominator, distinguish slots 36/45 from another eye slot, known 5-pixel errors, fraction/percent and values above 100%, unequal masks/image-macro aggregation, missing/zero anchors, all-missing targets, explicit units/order, overflow, origin declarations, non-square crop/padding, unclamped out-of-bounds points, EXIF edge transforms/inversion, legacy quantization and invalid inputs. A subprocess audit hook denies network connections and opening research artifact paths during the offline verifier, and the verifier rejects public output destinations.

| Check performed | Actual result |
|---|---|
| Initial targeted new tests | 34 passed; one additional percent-overflow case then added |
| Full preserved `.venv` suite | **319 passed, five expected optional Torch-module skips** |
| Full `.venv-baseline` suite | **407 passed**, no skips |
| Ruff lint / formatting | Passed / **57 Python files already formatted** |
| Both environment dependency checks | Passed |
| Offline verifier checks | **9/9 passed**, no optional image/numeric/model modules loaded |
| Existing locked inventory | **10** pins plus editable project in `.venv`; **19** plus project in baseline, checked by verifier subprocesses |
| Synthetic metadata/group CLI | Exit 0; expected unknown-identity limitation retained |

Full baseline tests include the already-existing manufactured-model training and temporary synthetic-checkpoint tests. No real-image/model/checkpoint artifact was an input. The new verifier itself performs no model, image or checkpoint operation. No packages or system settings were installed/changed. Historical environment records and locks are preserved.

All nine verifier checks passed: synthetic NME fraction 1 / percent 100 for a 5-pixel offset and 5-pixel denominator; 68 eligible points; missing anchor leaves NME undefined while retaining 67 eligible points; expected crop/padding coordinates; explicit source-origin conversion; legacy square mapping/truncation; and round-trip error **`4.440892098500626e-16` source pixels**, below the predeclared `1e-10` geometry tolerance. These are manufactured arithmetic results, **not InfAnFace model performance**. Initial style findings, including a test lambda, were corrected before final lint; there was no final test or verifier failure.

## Exact commands and preservation

From the project root, using existing environments only:

```sh
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_infanface_conventions.py
.venv/bin/python -m pytest -q -p no:cacheprovider
.venv-baseline/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/python -m ruff check --no-cache .
.venv/bin/python -m ruff format --no-cache --check .
.venv/bin/python -m pip --disable-pip-version-check --no-cache-dir check
.venv-baseline/bin/python -m pip --disable-pip-version-check --no-cache-dir check
.venv/bin/python scripts/verify_infanface_conventions.py --output reports/private/infanface-conventions-verification-2026-11-16.json
.venv/bin/python scripts/verify_infanface_conventions.py --output reports/private/infanface-conventions-verification-2026-11-16-final.json
.venv/bin/python -m infant_movement audit tests/fixtures/synthetic_metadata.json
```

The final verifier record was refreshed after the plan's wording clarification to bind the final document hash; the original record is retained. The verifier's stdout is safe synthetic aggregate JSON. It can be rerun **without `--output`**, or with an unused ignored private JSON filename. Existing destinations, public paths and resolved symlink escapes are refused. It records exact environment versions, source registry/code/config/lock hashes and local Git state, with zero network operations and explicit flags for no real-data NME or historical-results change. A separate ignored software-check record retains actual full-suite/lint/format/dependency command outputs.

Starting HEAD remained `6aaea5f`; existing correspondence/provenance changes were already uncommitted. A pre-edit inventory hashed **116 existing public files** and recorded size/mtime/inode for **135 sealed ignored artifacts** without opening them. Only README/AGENTS from the original public inventory changed, as authorized; the other **114** hashes match. All 135 sealed stat records match. This is preservation evidence, not a new content-hash authentication of unopened raw images/weights/predictions. Earlier uncommitted source/config/test/protocol/report files are retained unchanged. No commit, push, upload, external contact, split or identity operation occurred.

Changed files: new `datasets/coordinate_conventions.py`, `evaluation/infanface_conventions.py`, `scripts/verify_infanface_conventions.py`, `tests/test_infanface_conventions.py`, public source-evidence JSON, this report and its plan; README/AGENTS updates. The original research plan and every historical report remain unchanged. Private software/verification records remain Git ignored.

## Next bounded technical proposal — not executed

Implement a **synthetic heatmap target-generation and coordinate-decoding adapter** using the now-explicit edge/normalization contract. Require declared heatmap resolution/stride, missingness/support masks, quantization policy and reversible mapping to source coordinates; test non-square dimensions and known subpixel peaks before connecting any model. This would prepare a future heatmap landmark baseline without acquiring assets or running real inference/training. It needs a separate scoped request. Real-data use of a new metric/model still requires a declared protocol, local slot/frame assumptions and the unresolved grouping limitations; none is authorized automatically by this milestone.
