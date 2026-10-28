# InfAnFace image loading and landmark preprocessing

**Date:** 28 October 2026. **Scope:** explicit local access to the existing archive, reversible image/landmark geometry, synthetic tests and separate aggregate local-data smoke checks. No model, weights, training loop, inference, new datasets, network access, contact, upload, push, PR or background job was introduced.

## Outcome

Implemented an extraction-free local ZIP image loader and explicit crop, uniform resize/padding, EXIF frame handling and raw-to-output/inverse affine transforms. Original annotation objects and all baseline raw/historical files remain unchanged. No identities, source groups, project partitions or clinical labels were assigned.

**209 synthetic tests passed**, with lint, format, dependency and installed-lock-version checks passing. No dependencies were added. The separate real-data smoke checks are:

| Check | Stored policy | Normalize EXIF policy |
|---|---:|---:|
| Referenced / decoded images | 410 / 410 | 410 / 410 |
| Processed images | 410 | 409 |
| Decode failures | 0 | 0 |
| Preprocessing rejections | 0 | 1 (invalid EXIF tag `0`) |
| Source-out-of-bounds landmarks among processed samples | 1 | 1 |
| Original annotation changes | 0 | 0 |
| Maximum round-trip error, source pixels | 6.821210263296962e-13 | 6.821210263296962e-13 |
| Command exit status | 0 | 1 (incomplete preprocessing coverage) |

Both runs explicitly declared the **unverified hypotheses** `source_index_base=0` and `annotation_frame="stored"`, with whole-raster crops and 256×256 output. Normalization was a separate check; it was not substituted as the chosen policy after inspecting results. The invalid-tag image remains in the source archive and official metadata. Its rejection was not converted to a permanent data exclusion or hidden identity/split assignment.

Round-trip accuracy validates invertible arithmetic under the declared hypotheses. It **does not verify anatomical alignment, indexing origin, annotation raster frame, patient/video independence or model accuracy**. No source-image overlay/alignment adjudication was claimed in this milestone.

## Files and architectural choices

- `datasets/landmark_geometry.py`: decoder-independent immutable affine matrices, composition/inversion, explicit EXIF edge geometry, crop and exact-aspect letterbox geometry, missing-point preservation and bounds flags.
- `datasets/infanface_images.py`: context-managed archive integrity verification, validated annotation membership, bounded image reads/decoding, owned RGB rasters and `preprocess_face`. Pillow is imported only for image operations. The original annotation reader is unchanged.
- `scripts/smoke_infanface_preprocessing.py`: explicit offline smoke checks with aggregate stdout and a new private JSON record; refuses public paths and overwriting. It saves no images, coordinates, identifiers, candidate edges or processed assets.
- `tests/test_infanface_images.py`: 59 new synthetic test cases covering local archive ownership/limits, known pixel-marker placement under all EXIF cases, inverse geometry, explicit hypotheses, crops, fractional padding, missing/out-of-bounds points, colour/depth handling, invalid inputs and smoke-command behavior.
- README, AGENTS and a new environment record document the current scope and reproducible working contract. The original research plan and historical reports were preserved.

Reusing the existing pinned Pillow avoids a new numeric/model framework for simple deterministic 2D geometry. RGB images and Python coordinate tuples are returned rather than implying a model-specific tensor range, channel normalization, stride, heatmap target or loss. These later choices need their own tested contract.

## Coordinate and raster contract

The working frame uses **pixel-edge coordinates**. Raster edges are `[0,width] × [0,height]`; pixel `(column,row)` has centre `(column+0.5,row+0.5)`. Bounds flags use the half-open footprint `[0,width) × [0,height)`. Size arguments are always `(width,height)`.

Raw annotation numbers are preserved verbatim. A caller must explicitly choose base 0 or 1: raw `(0,0)` or `(1,1)` respectively hypothesizes the first pixel centre. The conversion adds `0.5-base` on both axes. Both choices remain **UNVERIFIED source conventions**. Other possible source conventions are not claimed resolved by supporting these hypotheses. The historical reader's numerical `[0,width) × [0,height)` QC is unchanged; its raw-number mask is not defined as identical to the new assumed-frame footprint mask at fractional borders.

The caller must also declare `annotation_frame="stored"` or `"display"`. Display-frame annotations are converted back through inverse EXIF geometry when working with stored pixels. `orientation_policy="stored"` keeps pixels unchanged; `"normalize"` applies EXIF to both pixels and coordinates. Missing EXIF uses the identity operation but does not establish upright content. Invalid orientation tags are retained as metadata and refused whenever frame conversion/normalization is requested. Crops are always interpreted after the selected pixel orientation policy.

All eight standard EXIF operations, including raster mirrors, are supported. For these metadata corrections the landmark index map is **identity: `i → i` for all indices 0–67**. The same source vertex index moves with its associated raster point; no left/right semantic augmentation or index exchange is performed. Canonical/anatomical index meanings after reflection remain UNVERIFIED until the source's indexing semantics are established. Synthetic markers verify position and index retention. A future horizontal-flip augmentation requires a separately supported anatomical swap mapping and is not implemented here.

`crop_box` must be `None` for the full raster or integer half-open `(left,top,right,bottom)` edges with positive area inside the selected raster. Boxes outside the image or with fractional edges are rejected. No crop is derived from landmarks, and no detector is implemented. Caller-supplied boxes are explicit; any future ground-truth-derived crop protocol must be labeled **ground-truth-crop preprocessing**, separately from detector-backed performance.

For cropped size `(cw,ch)` and requested `(ow,oh)`, one exact scale is `s=min(ow/cw,oh/ch)`, with padding `((ow-s*cw)/2,(oh-s*ch)/2)`. Continuous padding can be fractional; it avoids anisotropic effective scales from independently rounded raster dimensions. BICUBIC inverse-affine resampling maps output pixel centres to crop coordinates using the same geometry as the landmarks. Crop translation, optional frame conversion, resize and padding compose into `forward.matrix`; `inverse.matrix` maps output edge coordinates back to the original raw annotation numbers. Fractional and out-of-bounds coordinates are retained.

The prepared result retains the original sample and separate `present`, `source_in_bounds`, `crop_in_bounds` and `output_in_bounds` flags. None of these establishes landmark visibility or loss eligibility. An out-of-crop point may fall in output padding, so output bounds alone are insufficient for supervision. Explicit missing points remain `None`; nonfinite or malformed values fail instead of becoming zero. The validated source CSV still requires all 68 finite pairs; its schema and original annotations were not rewritten.

Future model outputs must be expressed in the working edge frame before applying `inverse`. A model using integer pixel centres needs `+0.5` on both axes first. Heatmap decoding, stride, normalized tensor coordinates and target validity policy remain unimplemented.

## Image handling and integrity

The loader checks the already recorded SHA-256 before opening images and again after its context. It reads only exact image members referenced by the validated table, without extraction. Missing or duplicate members, encryption, symlinks, over-32-MiB members, over-25-million-pixel rasters, unsupported depth/modes and multiframe formats fail. Supported images are single-frame JPEG/PNG in the declared ordinary 8-bit/palette modes. CRC/decode errors fail rather than silently replacing content.

Loaded images stay in stored orientation until preprocessing explicitly chooses a policy. Transparency is composited over a fixed white background; outputs are RGB. Padding defaults to black and can be explicitly configured. No ICC colour-profile conversion is performed. Original EXIF/XMP metadata is cleared from new output rasters to prevent repeated orientation application or accidental metadata redistribution. Callers own and close loaded/prepared images; no internal cache or processed-image file is created.

Archive SHA-256 remained `086efbcdb3e7bc9b718c012678d263f74abe1df42e018bac3574b721ab9d1c57`. All **20 retained baseline files** matched their initial hashes: the raw files, original unassigned inventory, historical reports/environment files/locks and research plan. The plan's SHA-256 remains `ba5424d9e1ba1f41c6edf4c9a759ec5cd29cc01ea74c474153d277e8114532ad`. Local hashes establish preservation against the acquisition receipt, not independently authenticated publisher provenance.

Official membership remains **210 training / 200 test**. EXIF counts observed by both smoke checks were absent 345, tag 1 in 57, tag 0 in 1, tag 8 in 5 and tag 6 in 2. No sample-level evidence or recognizable imagery is included in this report.

## Exact working instructions and environment

Use the existing `.venv`; no installation is needed. CPython 3.13.3 and Pillow 13.0.0 match the existing hashed lock. All distribution pins in both existing locks matched installed versions; lock files and original environment records are unchanged. See [the preprocessing environment record](../environments/infanface-preprocessing.md).

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m pip --disable-pip-version-check --no-cache-dir check
.venv/bin/infant-movement audit tests/fixtures/synthetic_metadata.json
```

All commands passed: 209 tests with no skips, lint, 23 formatted Python files, no broken dependencies and the expected synthetic metadata warning for unknown identity. Early draft lint formatting issues were corrected before final verification; there are no unresolved synthetic/lint failures.

To rerun local-data checks, from the repository root choose **new unused** private JSON filenames:

```sh
.venv/bin/python scripts/smoke_infanface_preprocessing.py --archive data/raw/infanface/InfAnFace.zip --receipt data/raw/infanface/acquisition-receipt-2026-10-27.json --source-index-base 0 --annotation-frame stored --orientation-policy stored --output-size 256 256 --output reports/private/infanface-preprocessing-stored-rerun.json
.venv/bin/python scripts/smoke_infanface_preprocessing.py --archive data/raw/infanface/InfAnFace.zip --receipt data/raw/infanface/acquisition-receipt-2026-10-27.json --source-index-base 0 --annotation-frame stored --orientation-policy normalize --output-size 256 256 --output reports/private/infanface-preprocessing-normalize-rerun.json
```

The first completed run returned 0. The second returned 1 because normalization explicitly rejected the invalid EXIF tag; the aggregate record was still written. Exit 1 also covers decode failures, annotation changes or round-trip error above `1e-8` pixels; inspect the aggregate fields. Exit 2 denotes rejected arguments/input/output. `--crop-box LEFT TOP RIGHT BOTTOM` applies one explicitly supplied box to every checked sample and may reject images with incompatible dimensions.

Original smoke records use `reports/private/infanface-preprocessing-{stored,normalize}-2026-10-28.json`. They and all sample-level evidence, raw/processed images, manifests and caches remain Git ignored. No image, per-sample coordinate or source ID is emitted by the smoke command. The smoke uses official test images only for fixed loading/geometry QC; it fits no preprocessing, model hyperparameter or exclusion threshold.

## Limitations and first-baseline prerequisites

- Source indexing origin, stored-versus-display annotation frame and anatomical alignment remain **UNVERIFIED**. A transform can be perfectly invertible under a wrong assumption. Obtain evidence and privately adjudicate alignment before freezing E1 input/target geometry.
- Resolve how the invalid EXIF tag and existing out-of-bounds annotation will be handled in a predeclared coverage/validity protocol. Keep originals unchanged and disclose any eventual exclusions; do not silently clamp or declare an annotation invisible.
- Current whole-image/caller-box processing is not a validated face detector. Ground-truth crops, detector crops and full-image localization must have distinct evaluation protocols.
- BICUBIC affine downsampling has no separate scale-aware antialias prefilter. It is a declared geometry implementation, not verified parity with HRNet preprocessing; strong minification may lose detail/alias. Review parity/antialiasing before a model benchmark.
- Subject/video/source grouping remains unresolved. Preserve official membership; no independent validation split was constructed or certified. Resolve corroborated relationships and approve a defensible evaluation protocol before training. Any eventual official image-level benchmark must disclose unresolved independence.
- Before the first model baseline, additionally review model source/checkpoint licensing, acquisition authorization and training provenance; establish a minimal pinned model environment; implement and test model-specific tensor/heatmap/metric conventions, including NME denominator and coverage. None of that work began here.

The next bounded milestone should be a **coordinate/alignment and E1 baseline-readiness review**, with evidence for source conventions and a frozen preprocessing/evaluation contract. Stop after this preprocessing implementation and verification. This project remains exploratory movement/geometry research, not seizure diagnosis or a clinically validated system.
