# InfAnFace: E0 access, licensing, and provenance desk review

**Review date:** 27 October 2026, Europe/Berlin. **Scope:** public documentation for one candidate dataset. **Decision:** desk review complete; acquisition readiness is **CONDITIONAL**, not blanket clearance for research use, redistribution, or training. No acquisition is authorized by this report.

The published noncommercial research purpose appears consistent with this project's exploratory landmark research. That is a scope assessment, not a finding that all underlying media permissions are resolved. Resolve the local-use and original-media questions below before proposing acquisition under the repository's data-rights rule. Unknown subject/source mappings independently limit evaluation claims; they are not evidence that the public download route is unavailable.

## Evidence register

**VERIFIED-DOC** means directly supported by documentation read on the review date. It does not mean verified against dataset contents. **UNVERIFIED** identifies missing or untested evidence. **PROPOSED** identifies project decisions, not dataset facts.

| Ref | Primary source | What was inspected |
|---|---|---|
| S1 | [Official repository README](https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking) | Dataset terms, access pointers, annotations, suggested partitions, model instructions |
| S2 | [Author-hosted directory index](https://coe.northeastern.edu/Research/AClab/InfAnFace/) | Listed documentation, archive name, file sizes and modification dates; no archive or image link followed |
| S3 | [Author-hosted dataset README](https://coe.northeastern.edu/Research/AClab/InfAnFace/readme.md) | Dataset-specific documentation and terms, read as text; no annotation table fetched |
| S4 | [Repository code license](https://raw.githubusercontent.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking/master/license) | MIT text for software/documentation; copyright notice names Sarah Ostadabbas, 2022 |
| S5 | [Paper, arXiv v3](https://arxiv.org/html/2110.08935v3) | Dataset construction, annotation conventions, split rationale and evaluation definitions |
| S6 | [GitHub revision metadata](https://api.github.com/repos/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking/commits/master) | Observed repository revision, not a dataset-release identifier |

The web reader could not parse S3's Markdown content type or retrieve S6; a bounded direct HTTPS text request succeeded for each. No authentication or maintainer contact was used. Mutable URLs can change. No independent archival snapshot of these pages was created.

## Access and release identity

**VERIFIED-DOC:** the author-hosted index lists `InfAnFace.zip` as 256M with modification time 2022-06-03 10:03; it also lists `labels.csv`, `readme.md`, and an image directory. These are listing observations, not an archive inspection or exact byte count. [S2](https://coe.northeastern.edu/Research/AClab/InfAnFace/)

**VERIFIED-DOC:** S6 returned repository commit `329abb24a858e3591d2b63f3bc0f3f5f41ac0b56`, dated 2026-01-09, with message “Update readme.md.” This identifies code/documentation history, not the hosted ZIP. The repository's named branch is mutable. [S6](https://api.github.com/repos/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking/commits/master)

**UNVERIFIED:** a dataset version/tag or DOI tied to these exact hosted bytes; publisher checksum; archive integrity and contents; whether direct archive access succeeds; whether bundled terms differ; correspondence of repository tables to the hosted table. No archive, CSV, image, or checkpoint request was issued. The paper is explicitly pinned to [v3, revised 27 May 2022](https://arxiv.org/abs/2110.08935v3); the paper version is not a data version.

**PROPOSED:** after separately authorized acquisition, record origin, receipt date, actual byte count and SHA-256 before extraction. Preserve the original archive and bundled documentation privately. Use a project receipt identifier transparently described as local, not an invented author release. Reconcile all sources before declaring an adapter-ready release.

## Rights: keep artifacts separate

| Artifact / use | Evidence and status | Project disposition |
|---|---|---|
| Dataset images and annotations | **VERIFIED-DOC:** S1/S3 describe noncommercial research, teaching, publication and experimentation, exclude commercial/litigation-related purposes, and state the data are not intended for diagnostic/product use | Review this exact purpose and any bundled conditions before acquisition/use; do not call this a permissive standard dataset license |
| Repository software | **VERIFIED-DOC:** S4 is MIT; it requires retention of its notice and license for covered copies | Applies to covered software; third-party components retain their own terms |
| Original photos / YouTube stills | **UNVERIFIED:** per-image ownership, local-use permissions, consent documentation and redistribution basis were not established by the inspected pages | Do not infer these from Google indexing, public video access, or the code license |
| Published paper | S5 labels the paper CC BY-NC-ND 4.0 | Paper licensing does not grant dataset or checkpoint rights |
| Pretrained checkpoints | **UNVERIFIED:** artifact-specific grant, exact files, hashes and training membership were not inspected | No download, use or weight-release clearance |
| Public examples, mirrors, derived features or trained weights | **UNVERIFIED:** permissions for these specific outputs | No redistribution clearance follows from this desk review |

Dataset conditions: [S3](https://coe.northeastern.edu/Research/AClab/InfAnFace/readme.md). Code terms: [S4](https://raw.githubusercontent.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking/master/license). Paper notice: [S5](https://arxiv.org/html/2110.08935v3).

No external communication has been sent. If existing documentation cannot resolve local-use or output permissions, prepare focused questions for the maintainers; sending them requires separate user authorization. Lack of publicly inspected evidence is not a claim that permission or consent does not exist.

## Annotations and documented partitions

**VERIFIED-DOC:** the dataset README describes 410 face images, 68 Multi-PIE landmark locations, and binary turned, occluded, expressive and tilted attributes in `labels.csv`. Its five batches are `ads` (104), `google` (100), `google2` (51), `youtube` (100), and `youtube2` (55). Sources include advertisement/video stills and Google-sourced photos. [S3](https://coe.northeastern.edu/Research/AClab/InfAnFace/readme.md)

**VERIFIED-DOC:** the repository recommends Train = `ads` + `google2` + `youtube2` (210), Test = `google` + `youtube` (200). Test contains Common (80; all four attributes false) and Challenging (120; at least one true). A separate infant validation partition is not described in that section. Do not mistake the repository's `300-W` validation files for InfAnFace validation. [S1](https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking)

**Discrepancy:** paper section III states 105 advertisement images, alongside 51 and 55 other training images, while also reporting 210 training images. Those numbers sum to 211. S3 instead states 104, which sums to 210. Actual file/row counts and the reason for the discrepancy are **UNVERIFIED**. Do not silently discard or invent a record to reconcile the sources. [S5](https://arxiv.org/html/2110.08935v3), [S3](https://coe.northeastern.edu/Research/AClab/InfAnFace/readme.md)

The paper documents a minimal landmark bounding box and a visible-face convention for self-obscured landmarks, with human annotations assisted by annotation tools/model initializations. The image-level occlusion flag cannot serve as a per-landmark visibility mask; separate visibility fields remain UNVERIFIED. For benchmark comparison, its interocular denominator uses outer eye corners; its alternative box denominator is the geometric mean of width and height. The plan's box-diagonal fallback would be a distinct project metric, not that published box metric. [S5, sections III and IV-B](https://arxiv.org/html/2110.08935v3)

**UNVERIFIED without data inspection:** exact column names/order, indexing origin, coordinate precision, image dimensions and transformations, missing-value codes, row/file correspondence, and per-landmark validity fields. Do not invent an executable adapter from prose alone. This resource supports a spatial-landmark experiment; no inspected evidence establishes temporal expression classes or confirmed seizure labels.

## Provenance and leakage assessment

The documented partition rationale concerns separately collected/annotated batches, not a demonstrated patient-disjoint registry. **UNVERIFIED:** unique infant count, complete original-video IDs, repeated infants across batches, reuploads, near duplicates, cross-dataset overlap, and upstream checkpoint training membership. No patient- or video-independence certification is justified by this review. [S5, section III](https://arxiv.org/html/2110.08935v3)

**PROPOSED application of the existing M1 contract:**

- Keep uncertain identity as `identity_status="unknown"` and `subject_id=null`; require actual evidence before assigning a known subject. Do not identify infants from faces.
- Keep unresolved `source_video_id`, frame index, and timestamps null. A still image does not supply a measured video timeline.
- Preserve author partition membership as `official_split`. Do not generate a validation split or silently promote it to an audited project partition.
- Establish conservative source/duplicate groups from documented relationships before auditing. Do not create one group per image merely to make checks pass; a collection batch is not an infant ID.
- Maintain an unassigned private inventory outside the M1 audit schema until release/grouping evidence and proposed project partitions are available. Required M1 identifiers cannot be filled with the string `UNVERIFIED`; use that marker in the evidence record, not as a fabricated identifier.
- Retain sample-level provenance and identity maps privately. A shared registry can detect overlap with another collection without merging supervision.

**PROPOSED evaluation gate:** after a separately authorized content audit, keep the official test sealed for tuning, derive validation only from eligible training groups, and quarantine any known train/test source or subject conflicts. If source grouping cannot be established, defer video-disjoint claims; if identity cannot be established, defer patient-disjoint claims. A passing M1 key-consistency check cannot supply either missing fact.

## Readiness decision and unresolved questions

| Decision | Status after this review |
|---|---|
| Select InfAnFace for further E0 investigation | **SUPPORTED:** appropriate candidate for the plan's independent spatial E1 experiment |
| Find an official access route | **VERIFIED-DOC:** hosted listing is reachable; archive access itself remains untested |
| Local noncommercial acquisition/inspection | **CONDITIONAL:** published purpose appears aligned; settle the local-use/underlying-media questions and seek explicit acquisition authorization before proceeding |
| Research training or inference | **NOT READY:** no content/schema audit, grouping audit or artifact-use review has occurred |
| Public media/derived-artifact redistribution | **NOT CLEARED:** output-specific rights remain UNVERIFIED |
| Patient/video-disjoint evaluation or clinical claims | **NOT ESTABLISHED:** no real-data audit or clinical supervision evidence |

Prioritize these questions before another milestone:

1. Do the posted dataset terms cover the intended local, independent noncommercial analysis of supplied images and annotations, and are additional conditions required for underlying media or any later shared outputs?
2. Which hosted archive/table is authoritative, is a publisher hash/version available, and how should the advertisement-count discrepancy be reconciled?
3. Are source-video/group mappings or repeated-subject references available without inferring identity? Which unknowns must remain unresolved?
4. Before any later pretrained-model use, what artifact-specific permissions and training membership are documented?

Archive hashes, row counts and duplicate discoveries are **acquisition-time checks**, not evidence that can be fabricated during a documentation-only review. Missing identities gate claims; they do not justify assigning surrogate patients. No new data inspection or model milestone begins automatically.

## Completion and validation scope

Deliverables are this public-source evidence record and a README status/link update. The research plan, M1 environment record, package, tests, dependency lock and original directory structure remain unchanged. No real metadata manifest or dataset configuration was generated. No datasets, labels, images, weights or additional Python dependencies were acquired; no models, training, inference, uploads, remote repository or background jobs were introduced.

For these documentation-only changes, verify local Markdown targets, file integrity and Git scope. The previous M1 verification remains historical evidence; this desk review does not rerun training checks or establish real-data audit success. Stop at this review boundary.
