# InfAnFace local duplicate and provenance audit

**Date:** 27 October 2026. **Scope:** only the already acquired local InfAnFace archive. No additional datasets/media, weights, models, training, inference, uploads, external contact, remote repository or background jobs were introduced. No project splits were created.

## Outcome

All 410 annotated images decoded. No exact decoded-pixel duplicate group was found under either declared pixel representation. Fixed-threshold screening of all 83,845 unordered pairs generated 23 candidates, all within the official training partition. All 23 were visually investigated for image-content correspondence:

| Review result | Pairs | Interpretation |
|---|---:|---|
| Plausible shared scene with distinct frame content | 1 | Corresponding background, garments and arrangement with relative pose changes; possible nearby frames, not a confirmed source-video mapping |
| No image-duplicate evidence beyond a white-background layout | 14 | Different scene content; perceptual-hash collision does not establish common origin |
| Different scene content with letterboxing/subtitle presentation | 8 | Layout drives similarity; shared presentation is at most an unresolved source clue |
| Confirmed transformed copies of the same image | 0 | None established by this review |

This was a single assistant visual-content review, not independent human adjudication. It made **no infant identity judgments**. The shared-scene pair remains a candidate requiring corroboration; other review outcomes do not establish different subjects or independent videos. No candidate was automatically converted to a duplicate, source or subject group.

**Split recommendation:** current evidence does not support a certified patient- or video-independent project split. Preserve the official membership as benchmark metadata, retain the within-training scene-related pair as a private grouping caution, and defer independent project split creation until source relationships can be supported. Absence of cross-partition candidates is not a clearance of the official split.

## Inputs, preservation and environment

The archive was checked against its existing acquisition receipt before decoding and rechecked afterward:

```text
data/raw/infanface/InfAnFace.zip
268,135,021 bytes
SHA-256 086efbcdb3e7bc9b718c012678d263f74abe1df42e018bac3574b721ab9d1c57
```

Baseline/final hashes also verify that all nine pre-existing files under `data/raw/`, including directory markers, remain unchanged. The original research plan, historical public reports, historical M1 environment/lock and existing unassigned sample inventory are preserved. A local archive digest establishes consistency with received bytes, not an independently published checksum.

Only **Pillow 13.0.0** was added to the existing CPython 3.13.3 arm64 `.venv`, as an optional `image-audit` extra. No core runtime dependency or additional numeric/model package was added. The selected native wheel was downloaded from PyPI, compared with its published SHA-256, then installed offline with `--no-deps --require-hashes --only-binary=:all:`. A synthetic PNG encode/decode check passed.

```text
pillow-13.0.0-cp313-cp313-macosx_11_0_arm64.whl
SHA-256 f11a3c1358bb3c68012a7b1f41b223c3db9f7bfde90b481df028cfe1c3363313
```

The new [image-audit lock](../environments/macos-arm64-py313-image-audit.lock) includes the unchanged M1 tool lock plus this exact decoder wheel. This lock is verified for this interpreter/platform only. [Official installation documentation](https://pillow.readthedocs.io/en/stable/installation/basic-installation.html), [PyPI release](https://pypi.org/project/pillow/13.0.0/).

## Reproducible image method

The code in `evaluation/image_duplicates.py` implements **`whole-image-v1`**. Thresholds were fixed and tested synthetically before the real audit; they were not tuned on the real candidate results or model outcomes. No learned representations, face detector, facial recognition, landmark-based face crop, or identity embedding was used.

1. Read image members referenced by the validated annotation table, directly from the ZIP without extraction. Permit JPEG/PNG single-frame images, at most 25 million pixels, with an explicit supported 8-bit/palette colour-mode set. Corrupt, oversized, animated and unsupported-depth images fail rather than silently changing representation. Decode failures count against coverage.
2. Hash dimensions plus **RGBA8 decoded pixels**, both as stored and after EXIF orientation. The byte preamble includes the raster dimensions; same bytes arranged in different shapes are not equal images. No ICC colour-profile conversion is applied. Pixel equality is defined by this representation, not by file bytes or all possible colour-managed appearances. Alpha is retained for exact equality. [EXIF orientation operation](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html#PIL.ImageOps.exif_transpose).
3. For candidate screening only, composite transparency over white and form oriented RGB views: full frame, horizontal mirror, central 80% crop, and its horizontal mirror. Crop bounds use integer 10% margins. This changes only temporary views, never raw images or annotation coordinates.
4. For each view, compute 64-bit **dHash** from a LANCZOS-resized 9×8 grayscale image using strict left-pixel > right-pixel comparisons. Compute 64-bit **aHash** from an 8×8 grayscale image using strict pixel > mean comparisons. Hash equality is not confirmation: uniform backgrounds and black bars can collide.
5. Compare every unordered sample pair over all 16 view combinations. A pair becomes a candidate if the **same view combination** passes **dHash Hamming distance ≤6 AND aHash Hamming distance ≤8**. Exact oriented-pixel equality is included regardless. The best qualifying view is selected lexicographically by `(dHash+aHash, RGB-thumbnail MAE, dHash, aHash, left-view name, right-view name)`.
6. Colour MAE uses 16×16 RGB thumbnails and normalization by `768×255`; it ranks qualifying views and is not a confirmation threshold. Aspect ratios are resized to the fixed hash/thumbnail shapes, so matching silhouettes/layouts can be false candidates.
7. Retain every candidate privately. Review full-image contact sheets, with a larger pair view where needed. Assess matching scene content and geometry; do not compare faces to identify a child. Record explicit evidence, uncertainty, review method and a decision for every pair.

All candidate-generation operations are deterministic under the locked decoder and code; no random sampling or fitted thresholds occur. Visual judgments are qualitative and may vary across reviewers. Reproducible features do not make reviewer judgments ground truth.

## Image findings

| Check | Result |
|---|---:|
| Referenced images / successful decodes | 410 / 410 |
| Decode failures | 0 |
| Exact stored-RGBA8 pixel groups | 0 |
| Exact EXIF-oriented-RGBA8 pixel groups | 0 |
| Pair comparisons | 83,845 |
| Candidate pairs | 23 |
| Candidates crossing official train/test | 0 |
| Candidate pairs within `ads` / within `youtube2` | 15 / 8 |
| Candidate pairs investigated | 23 |

Decoded source modes were RGB (305), RGBA (104), and grayscale L (1). Seven images have EXIF orientation 6 or 8; one has a nonstandard orientation value 0, which is retained in private evidence. These observations concern audit views, not a new training-preprocessing decision. Future coordinate transforms must explicitly account for orientation if applied; the annotation reader and the previously flagged out-of-bounds landmark were not altered.

The screen has no measured recall on independently labeled infant duplicates. It can miss large/off-center crops, overlays, substantial appearance changes, arbitrary rotations or different frames from one video. The 22 layout-driven candidates demonstrate why a hash match is a review request, not duplicate confirmation. Rejected image matches do not rule out a common compilation, recording or subject.

## Local source and provenance review

Only available local evidence was inspected: annotation batch/filename fields, the retained archive documentation, selected EXIF text/timestamps, and whitelisted XMP source/document-reference properties. No media URL was fetched, no external page or dataset was scraped, and no person was contacted. On-image personal names were not transcribed or used as identity keys. GPS metadata values were not inspected.

The source script implements **`local-source-metadata-v1`**:

- Extract bounded XMP source/relation/identifier and document-lineage fields in declared Adobe/DC namespaces. Reject DOCTYPE/entity declarations, malformed XML and oversized references. No URLs or external entities are resolved.
- Inspect EXIF DateTimeOriginal from its nested EXIF IFD where available, separately from root-level DateTime. These are metadata timestamps, not verified recording times.
- Compare local document-lineage tokens within and across eligible document-reference fields to generate source candidates. A shared editor, timestamp or metadata template is never equated with a shared video or infant.

107 images expose whitelisted XMP source/document references, and 19 expose nested EXIF DateTimeOriginal strings. No cross-sample document-lineage candidate pair was found, including no cross-partition one. No HTTP source/video URL was found in the inspected reference fields. **Two XMP blocks could not be parsed and remain UNVERIFIED**; other uninspected metadata may exist. Shared software strings and presentation templates are non-specific clues, not authoritative provenance.

No authoritative subject or source-video registry was established. All prior `subject_id`, `source_video_id`, `source_group_id` and `project_split` nulls remain unchanged. New review records also make no identity or project assignment. The full patient/video provenance component of E0 remains unresolved.

## Working instructions

Run from the repository root using the existing `.venv`. To reproduce the optional decoder environment on the same platform, the following explicitly downloads only hash-locked Python wheels as needed; it does not acquire research assets:

```sh
.venv/bin/python -m pip --disable-pip-version-check --no-cache-dir install --only-binary=:all: --require-hashes -r environments/macos-arm64-py313-image-audit.lock
.venv/bin/python -m pip --disable-pip-version-check --no-cache-dir install --no-index --no-build-isolation --no-deps -e .
```

Choose **new** private output names for reruns; the commands refuse overwriting. Neither command downloads files or creates splits. Stdout is aggregate; JSON contains private per-sample evidence and remains under `reports/private`:

```sh
.venv/bin/python -m infant_movement duplicate-audit --archive data/raw/infanface/InfAnFace.zip --receipt data/raw/infanface/acquisition-receipt-2026-10-27.json --output reports/private/infanface-duplicates-rerun.json
.venv/bin/python scripts/audit_infanface_sources.py --archive data/raw/infanface/InfAnFace.zip --audit-report reports/private/infanface-duplicates-rerun.json --output reports/private/infanface-sources-rerun.json
```

The first command returns 0 for complete image decoding, 1 for incomplete decode coverage, or 2 for a rejected invocation/input. The source command records unparsed XMP blocks in its output and still returns 0 when the metadata inspection completes; inspect `xml_failure_count`, not just the process status. The audit reads all referenced images for data QC, including official test images, but fits no model or preprocessing and selects no model hyperparameters.

Existing evidence is private and Git ignored:

```text
reports/private/infanface-duplicate-audit-2026-10-27.json
reports/private/infanface-source-audit-2026-10-27.json
reports/private/infanface-duplicate-review-2026-10-27/
reports/private/infanface-duplicate-method-2026-10-27.json
reports/private/infanface-pillow-receipt-2026-10-27.json
```

The review directory contains private contact sheets, the larger review view, a completed decision log, and intermediate metadata evidence. No sample-level image/hash, candidate edge or identity field is included in this public report. The method record retains code/lock hashes and preservation checks; the dependency receipt retains the selected wheel URL/hash. No original manifest was rewritten.

## Verification and limits

**150 synthetic tests passed**, with no skipped tests in the image-audit environment: the previous 121 plus 29 new duplicate/provenance tests. Lint, format and dependency checks passed. Checks cover pixel equality across lossless encodings/metadata changes, raster dimensions, EXIF normalization, JPEG/resize/mirror/crop candidates, deliberate uniform-background collisions, inclusive thresholds, corrupt/multiframe/high-depth rejection, checksum failure before parsing, private path/symlink handling, no overwriting, a full synthetic archive/CLI/source-script path, and bounded/entity-free XMP parsing. No infant imagery or dataset-dependent fixtures are embedded in tests.

Executing both documented rerun commands into new private paths produced **byte-identical machine reports** to the initial image/source audits. The archive/raw-file hashes also remained unchanged. This checks deterministic machine processing, not inter-reviewer agreement; the qualitative visual judgments were not regenerated.

The base package still has zero required runtime dependencies. Audit-specific tests are explicitly optional without Pillow; the historical M1-only environment does not constitute a complete duplicate-audit run. This phase's full test count uses the new image-audit lock. No training, inference, model accuracy, clinical validity or patient-independent generalization has been evaluated.

## Proposed next step and split recommendation

Stop after this audit. The next bounded step should be independent adjudication of the private shared-scene/source clues and resolution of a documented source grouping basis, together with explicit handling of the existing coordinate-quality/orientation issues before E1 preprocessing.

Keep the original train/test labels immutable. A future benchmark reproduction could only be described as an **image-level official split with unresolved subject/video independence**, not patient- or video-held-out validation. Do not claim a defensible independent validation partition by randomly dividing the 210 training images or assigning one patient/source group per file. Any corroborated related-image/source component must stay together in a later split, but neither its IDs nor assignments are created here. Patient/video-level claims remain deferred until adequate provenance evidence exists.
