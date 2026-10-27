# InfAnFace E0 acquisition and annotation audit

**Date:** 27 October 2026. **Scope:** authorized private acquisition, integrity/schema/provenance inspection, and a dependency-free annotation reader. No model, training, inference, upload, external message, or remote repository was introduced.

## Decision and authorization

The user explicitly authorized the acquisition-only E0 audit, then authorized routine local follow-up with a written plan before execution. A plan was provided before acquisition and before the annotation-reader phase. The earlier [desk review](infanface-e0-desk-review-2026-10-27.md) remains an unchanged historical record of what was known before download.

For this private noncommercial inspection, the author-posted research terms were retained as the stated use basis. This does not certify the underlying media rights chain, consent records, clinical suitability, or redistribution rights; those remain **UNVERIFIED**. The bundled README matches the fetched author-hosted README byte for byte. It does not supply a new grant for public media or derivative-artifact redistribution. No external clarification was requested or received.

**Outcome:** archive receipt/integrity and annotation-schema checks are complete. A minimal local annotation reader is verified. **A patient/video-independent project split is not ready**, and no project train/validation/test assignments were created. The full provenance component of E0 remains incomplete.

## Acquisition evidence

| Item | Recorded observation |
|---|---|
| Official access page | [Author-hosted InfAnFace directory](https://coe.northeastern.edu/Research/AClab/InfAnFace/) |
| Retrieved archive URL | `https://coe.northeastern.edu/Research/AClab/InfAnFace/InfAnFace.zip` |
| Local raw archive | `data/raw/infanface/InfAnFace.zip` — Git ignored |
| Received bytes | 268,135,021 |
| Local SHA-256 | `086efbcdb3e7bc9b718c012678d263f74abe1df42e018bac3574b721ab9d1c57` |
| Receipt | `data/raw/infanface/acquisition-receipt-2026-10-27.json` — private/ignored, includes UTC timestamp and HTTP response headers |
| Author documentation | `author-readme-received-2026-10-27.md` and `author-index-received-2026-10-27.html`, retained beside the raw ZIP |
| Publisher release identifier / checksum | **UNVERIFIED**; the local digest identifies received bytes, not independent publisher authentication |

No archive files were extracted or executed. Paths, entry duplication, encryption flags, member sizes, and symlinks were checked before semantic inspection. Every file stream was read successfully with ZIP CRC validation. These checks establish transport/container consistency, not that arbitrary content is universally safe or scientifically valid.

The archive contains 845 entries: eight directories and 837 files. After excluding macOS resource-fork sidecars, the image payload is 210 JPEGs plus 200 PNGs. One real annotation CSV and one real README were parsed. OS metadata was retained in the original archive but excluded from semantic image/annotation counts. An initial inventory attempt stopped on a binary `._labels.csv` resource fork; the completed pass distinguishes these sidecars from real UTF-8 text. No text replacement or encoding fallback was needed for the real CSV/README.

## Observed annotation schema

The actual table is `infanface-public/labels.csv` with **410 rows and 144 columns**:

- `image-set`, `filename`, `partition`, `subpartition`;
- binary image attributes `turned`, `occluded`, `expressive`, `tilted`, stored as numeric `0.0` / `1.0`;
- 136 coordinate values: `gt-x0`, `gt-y0`, through `gt-x67`, `gt-y67`.

All 410 rows map to distinct existing image paths. There are no unreferenced semantic image files, repeated image references, malformed rows, or nonfinite coordinate values. Coordinates include fractional values and must not be truncated to integers.

| Batch | Image / annotation rows | Official partition |
|---|---:|---|
| `ads` | 104 | `infanface-train` |
| `google2` | 51 | `infanface-train` |
| `youtube2` | 55 | `infanface-train` |
| `google` | 100 | `infanface-test` |
| `youtube` | 100 | `infanface-test` |
| Total | 410 | 210 train / 200 test |

The downloaded payload therefore contains **104 advertisement images**. This resolves the actual count for this received archive; it does not establish why the paper's prose said 105. The original desk-review discrepancy remains documented rather than rewritten.

Official test subpartitions contain 80 Common and 120 Challenging images. Every Common row has all four attributes false, and every Challenging row has at least one true. Official labels are preserved; they are not automatically promoted to audited project partitions. There is no validation partition in the received table.

## Geometry and quality findings

PNG/JPEG dimension headers were readable for all 410 referenced files. Widths range from 270 to 5,000 pixels and heights from 168 to 3,264 pixels. Image pixel decoding and visual annotation review were **not performed**; readable headers are not proof of complete image decodability.

Under the explicit geometric check `0 <= x < width` and `0 <= y < height`, **one landmark in one test image is outside the image bounds**. Its exact row, landmark and coordinates are retained privately. The reader preserves the source coordinate; it does not clamp, drop, relabel, or silently correct it. A later evaluation policy must handle this quality issue explicitly before model scoring.

**UNVERIFIED:** source coordinate-indexing origin, independently checked annotation accuracy, per-landmark visibility, subject identity, source-video mappings, and timestamps. The image-level occlusion attribute cannot substitute for a landmark-visibility mask. CSV landmark index names 0–67 do not establish whether pixel coordinates use zero- or one-based conventions. No crop centers/scales or bounding-box columns are present in this table; future crop derivation must be declared as project preprocessing.

## Duplicate and provenance findings

SHA-256 comparison of the 410 referenced image files found **zero exact-byte duplicate groups**, including zero crossing the official partitions. This does not detect re-encoded, resized, cropped, near-identical, or repeated-subject images. Pixel/perceptual near-duplicate analysis and cross-dataset overlap remain **UNVERIFIED**.

The annotation table provides no subject or source-video IDs. No supplied subject/source mapping was established from the inspected archive. There is no basis for calling its official partition patient-disjoint or video-disjoint. Filename patterns and collection batches were not used to infer infant identity.

The private inventory deliberately records `subject_id=null`, `source_video_id=null`, `source_group_id=null`, and `project_split=null`. It is an **unassigned inventory**, not an M1 grouped-split manifest: required M1 grouping/assignment fields have not been fabricated. No real-data “clean split audit” is claimed. This resource's landmark supervision remains separate from every other dataset.

## Local reader and verification

Added `src/infant_movement/datasets/infanface.py` and synthetic-only `tests/test_infanface.py`. The reader:

- reads an explicit local UTF-8 CSV or the exact table member of the observed ZIP layout, without networking, extraction, image decoding, or implicit downloads;
- validates the schema, binary attributes, finite coordinates, source batches, unique image references, and official partition/subpartition consistency;
- preserves fractional/out-of-bounds coordinates and returns a separate geometric bounds mask when dimensions are supplied;
- preserves unknown identity/grouping and leaves project assignments unset;
- rejects unsafe image path references, duplicate/oversized archive label members, and malformed tables without printing private filenames.

Verification completed in the existing Python 3.13.3 `.venv`:

| Check | Result |
|---|---|
| Full synthetic suite | **121 passed**: original 82 plus 39 reader checks |
| Ruff lint / format | Passed; 13 files formatted consistently |
| Dependency consistency | Passed; package still has zero runtime dependencies |
| Real-data annotation-reader smoke check | 410 rows; official counts preserved; all identity/project assignments unset; one geometry flag preserved |
| Raw archive SHA-256 recheck | Matches acquisition receipt |

Real-data smoke checks are separate from the synthetic suite; tests do not embed or require infant data. No tests of model accuracy, training, GPU behavior, landmark transformations, clinical classification, or pixel decoding are implied.

Private files, all Git ignored:

```text
manifests/infanface-unassigned-inventory-2026-10-27.json
reports/private/infanface-archive-inventory-2026-10-27.json
reports/private/infanface-initial-summary-2026-10-27.json
reports/private/infanface-metadata-audit-2026-10-27.json
reports/private/infanface-bundled-readme-2026-10-27.md
reports/private/infanface-acquisition-helper-2026-10-27.py
reports/private/infanface-metadata-helper-2026-10-27.py
reports/private/infanface-audit-method-2026-10-27.json
```

The research plan and historical M1 environment record remain unchanged. User authorization and current phase status are reflected in AGENTS/README. No new dependency lock, pretrained asset, training code, project split, Git commit, or remote was created.

The one-time audit helpers are retained privately with method hashes for reproducibility; they are not package entry points, background tasks, or automatically executed imports. The acquisition helper performs network access only if explicitly run. Review its recorded scope before reusing it.

## Next phase plan

The next useful work is provenance review before split creation: examine near-duplicate/source relationships with a documented method, retain uncertain relationships, and seek supplied source-group evidence where available. Any new image-decoding dependency should be isolated, pinned and tested before use. Do not turn one source file into one asserted patient or source video. If provenance cannot support independent groups, report that limitation and defer patient/video-generalization claims.

Before a later training phase, declare a validation protocol that the available grouping evidence actually supports, a treatment for the out-of-bounds annotation, crop/coordinate conventions and relevant permissions for the selected artifacts. Models, weights, training, inference, and clinical interpretation have not been initiated by this milestone. Each subsequent execution phase will be described before work begins, consistent with the user's standing authorization for routine local work.
