# InfAnFace provenance and validation-readiness review protocol

Written **16 November 2026, before evidence classification**. Review
`infanface-provenance-readiness-v1` is limited to the exact original ordered
32 usable official-training references in the completed pilot. Starting commit
is `6aaea5fa0e29299ea1c7471e4925a4a9d9e36155`, branch
`research/infanface-e0-audit`, with the correspondence milestone's uncommitted
changes already present. Preserve those changes and every historical report/run,
the original plan, annotations and official membership. No new experiment.

- **Permitted evidence:** already retained local reports/protocols, environment
  and access/terms documents, pilot selection/configuration/coverage/preparation/
  provenance/completion metadata, integrity manifests, prior training-only
  relationship/candidate decisions and local source-evidence metadata. Hash and
  cross-check available metadata references. Raw archive byte hashing is allowed;
  do not open its members. Do not open imagery, overlays, video, annotation tables,
  learned features, checkpoints or prediction files, from any partition. Where
  a retained JSON index spans partitions, use reference/membership fields only
  to filter to this cohort before examining evidence. Test-only records stay
  sealed; use historical public aggregates for any cross-partition limitation.
- **Frozen cohort:** verify selection/config/coverage digests and ordered reference
  digest against existing preparation/provenance and later frozen manifests.
  Require 32 distinct, originally usable, official-training references in their
  recorded order and unchanged configuration/annotation hashes/boxes/mask counts.
  Do not re-read coordinates or reconstruct tensors. Historical checkpoint,
  prediction and pixel integrity claims remain historical; this review will
  not reopen their payloads. Newly taken metadata snapshots prove preservation,
  not retrospective authenticity. Refuse overwrite or writing inside old runs.
- **Evidence classes:** `documented` requires an explicit supplied relationship
  or authoritative retained mapping linking the sample to the stated subject,
  recording, group or session, with traceable local evidence/hash and no unresolved
  contradiction. `provisional` means an explicit uncertain relationship hypothesis
  supported by retained evidence, not a verified key. `conflicting` means relevant
  retained sources disagree. Missing/unsupported evidence is `UNVERIFIED`, value
  null. Never infer an identity/recording from appearance, filenames or batch.
  Keep prior similarity/shared-scene hypotheses separately from subject/video
  assignments; a similarity candidate does not create a group or close identity.
- **Private matrix:** one row per frozen example, in order, with official
  membership/batch; nullable subject, recording, source-group and session evidence;
  class, local evidence references and SHA-256s; relevant earlier candidate
  decisions; unresolved grouping obstacles. Include relevant cohort-to-cohort and
  cohort-to-other-training candidate relationships privately. Do not disclose
  names, links, IDs or sample-level findings in public reports.
- **Readiness decision:** subject-disjoint validation requires supported mappings
  covering the cohort and resolution of possible repeat-subject relationships.
  Source/video-disjoint validation requires supported recording/source-group
  mappings and unresolved reupload/shared-source relationships addressed. A
  collection batch, image filename or duplicate screen is not an independent
  group. An explicit conservative proxy must be labeled as such and cannot
  establish patient independence. If mappings remain insufficient, choose
  **grouped validation unresolved; descriptive work only**, without generating
  partitions or treating absence of conflicts as evidence of independence.
- **Separate prerequisites:** inventory source coordinate/frame hypotheses,
  anatomical index evidence, metric normalizers and dataset/code/media/weight
  rights independently. Do not promote reversible arithmetic, old qualitative
  alignment, a code license or public access into anatomical verification or
  legal clearance. No seizure labels or clinical claim can be derived here.
- **Execution and records:** use existing `.venv`, no installation/network,
  no model imports/execution or model/training/image-processing test collection.
  Prefer existing utilities/documentation; any new metadata helper gets synthetic
  metadata tests, relevant metadata-only checks, lint/format and dependency check.
  New ignored `runs/infanface-provenance-review-2026-11-16/` will contain resolved
  config, starting Git/preservation record, evidence references/hashes, private
  matrix, aggregate readiness, command/check log, completion/failure status and
  output-integrity manifest. Recheck preserved metadata and initial user changes
  after completion; list sealed payloads as unexamined. Update README and write
  one dated aggregate report. Stop; another action needs authorization.

**Implementation correction, same date:** the first command stopped at lock
inventory preflight, before software subprocess checks or matrix generation. The
image-audit lock contains a local `-r` include of the nine-pin development lock;
the initial checker counted its one direct Pillow pin only. Resolve that retained
local include with cycle/path/conflict/unhashed-line rejection; verify all ten
unchanged pins plus editable project. Preserve the failed directory unchanged.
Complete this same metadata review in fresh
`runs/infanface-provenance-review-2026-11-16-verified/`. No dependency, cohort,
evidence category or readiness criterion changes.
