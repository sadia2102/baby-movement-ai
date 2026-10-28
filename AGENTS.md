# Repository instructions

## Scope and review boundary

This repository has completed M1 metadata tooling and an authorized InfAnFace E0 acquisition/schema audit with an independent annotation reader. Read `README.md`, `research_plan.md`, and the latest E0 report before changes. The research plan remains the original proposal; README and dated reports record implementation status.

On 27 October 2026 the user authorized routine local research follow-up and required a written plan before execution. Announce each concrete phase before working; do not request repeat scope confirmation for already authorized routine local work. The current completed phase covers official InfAnFace acquisition, private integrity/schema/provenance inspection and a dependency-free reader with synthetic tests. Models, weights, training and inference remain unimplemented; announce a distinct phase and check scientific/data prerequisites before starting them.

Keep downloads explicit and phase-scoped; never fetch assets implicitly during imports, tests or annotation reads. External messages, uploads, remote publication, destructive operations and security/credential changes require specific authorization. No background jobs are part of the current workflow. Tool-enforced sandbox approvals still apply.

The latest bounded request on 28 October 2026 authorizes local InfAnFace ZIP image loading and reversible landmark preprocessing, minimal necessary environment changes, synthetic verification, a private aggregate local-data smoke check and documentation. Stop after implementation and verification. Keep raw files, historical reports and the research plan unchanged; do not create project splits, obtain additional datasets/media or weights, implement models/training loops, train/infer, contact anyone, upload, push commits, create a PR or start background jobs. Treat similarity as review candidates, never infant identity evidence. Source indexing and annotation raster conventions remain UNVERIFIED; require explicit caller hypotheses and transform pixels/landmarks together for EXIF handling. This narrower phase boundary takes precedence over broader standing follow-up authorization.

## Scientific meaning

- This is exploratory movement-analysis research, not diagnosis or clinical monitoring.
- Never interpret normal/normative movement, neuromotor risk, GMA scores, pose estimates, expressiveness, or anomaly/forecast scores as clinically confirmed seizure labels.
- Do not produce seizure sensitivity, specificity, detection claims, or clinical probabilities from the current datasets.
- Keep InfAnFace, YouTube Infant Body Parsing, BabyPose, and Chambers supervision separate. Never concatenate their records into one supervised training dataset or synthesize absent labels.
- Fuse modalities only for the same verified recording and timestamps. Depth pose results do not establish RGB pose accuracy. Landmarks do not establish emotion or expression-class accuracy.
- Label proposed methods as proposals and unrun checks as unrun. Cite primary sources for dataset facts and record verification dates. Mark missing evidence **UNVERIFIED** rather than guessing.

## Data, licensing, and privacy

- Preserve immutable raw data outside version control. Separate dataset, code, pretrained-weight, and original-media rights; one license does not grant the others.
- Resolve unverified data rights and access conditions before acquisition/use. Do not redistribute infant imagery, video URLs, metadata, learned features, or weights without reviewing applicable permissions. For the authorized private InfAnFace inspection, the posted noncommercial research terms are retained as the stated use basis; the underlying-media rights chain and redistribution permissions remain UNVERIFIED.
- Use pseudonymous subject and source-video IDs. Keep identity maps and actual manifests private; do not publish recognizable examples or credentials.
- Do not use facial recognition to infer infant identity. Unknown identity remains unknown and limits generalization claims.

## Leakage prevention

- Split before frame extraction, windowing, augmentation, feature fitting, or pseudo-labeling. All recordings of a known infant and all derivatives of a source video belong to one partition.
- Audit repeated videos, reuploads, near duplicates, and cross-dataset overlap, particularly body parsing versus Chambers. Use a common private provenance registry without merging supervision.
- Fit learned preprocessing, model selection, and decision thresholds on training/validation only. Never train on unlabeled test data or tune against final-test metrics.
- Fine-tuned upstream models used in temporal experiments must exclude downstream held-out subjects/videos. Make training provenance part of every feature-cache key.
- Unknown subject identity is not proof of patient independence. If it cannot be resolved, report only audited source/video-disjoint performance or defer the patient-generalization claim.
- Preserve official splits as immutable reference metadata. If an official split violates grouping, use the clean project protocol for primary results and mark a separate benchmark reproduction as limited or skip it.

## Implementation conventions after authorization

- Use a `src/infant_movement` package, independent dataset adapters, ordinary PyTorch loops, explicit configuration, and local file-based run records. Keep notebook logic thin.
- No automatic downloads, cloud uploads, telemetry, or experiment trackers. Network access to assets must be explicit.
- Store timestamps, coordinate systems, transforms, label provenance, confidence, and valid/occluded masks. Missing labels are not zeros or negative-class examples.
- Prefer CPU-portable code with explicit `cpu`, `mps`, and later `cuda` selection. Report unsupported operations and any fallback. Avoid hard-coded CUDA and mandatory CUDA-only dependencies.
- Pin tested dependency versions, source revisions, and checkpoint hashes before a substantive experiment. Do not call an untested version range a validated environment.
- Record resolved config, git state, data/split hashes, seeds, environment, device, preprocessing, exclusions, training state, and metric definitions for each run.
- Synthetic tests should target coordinate inversion, left/right flips, mask palettes, depth precision, timestamp integrity, causal receptive fields, split intersection, masked losses, and metric arithmetic. No real infant data or large assets in tests.
- Run relevant checks after implementation changes; record what ran and what could not run. Do not add implementation-mirroring tests for documentation changes.
- Do not fabricate results, licenses, annotations, subject IDs, or compatibility claims. Never commit raw assets, caches, checkpoints, private manifests, environment secrets, or identifiable outputs.

## Completion reporting

Describe changed files, verification actually performed, unresolved evidence, and the next bounded phase. Keep unknown provenance explicit and do not advance scientific claims beyond available evidence. Preserve historical evidence records; record new results in a dated report. Explain the next execution plan before starting it under the user's standing authorization.
