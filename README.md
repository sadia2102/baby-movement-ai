# Infant Movement Analysis Using Deep Learning

An independent, exploratory PyTorch research project for measuring infant facial geometry and body movement over time. The long-term question is whether validated movement measurements could support research into movements potentially associated with epileptic seizures.

**This is not a medical diagnostic system.** None of the candidate datasets is established here as providing clinically confirmed seizure events. Normative movement, neuromotor risk, facial expressiveness, and seizure status are different concepts; no conversion between these labels is permitted.

## Current status

**M1 metadata implementation completed on 27 October 2026**, within the user's authorization for package setup, minimal dependencies, and synthetic metadata/split checks. The package has no runtime dependencies. Development and build tools are installed in an isolated local `.venv`; the tested versions and wheel hashes are recorded below. Local Git is initialized on `main`, with no remote configured.

**InfAnFace acquisition/schema audit and the first independent annotation reader completed on 27 October 2026.** Only the official InfAnFace archive was acquired for authorized private noncommercial E0 inspection. The package still has no runtime dependencies. No model weights, training, inference, uploads, or remote repository are involved. There are no scientific model results or audited patient/video-independent project splits; identity and source-video mappings remain UNVERIFIED. The reader preserves official metadata and fractional coordinates without assigning project partitions.

- [Detailed research plan](research_plan.md): dataset evidence, experiment protocols, metrics, leakage controls, dependency recommendations, hardware strategy, and weaknesses.
- [Contributor and agent rules](AGENTS.md): scope boundaries and reproducibility requirements.
- [InfAnFace acquisition and annotation audit](reports/infanface-e0-acquisition-audit-2026-10-27.md): receipt/hash, observed schema, quality findings, provenance limits, and 121-test verification.
- [InfAnFace duplicate/provenance audit](reports/infanface-duplicate-provenance-audit-2026-10-27.md): decoded pixels, near-duplicate candidates, local source evidence, limitations and 150-test verification.

The [InfAnFace desk review](reports/infanface-e0-desk-review-2026-10-27.md) and acquisition audit remain unchanged historical records. The subsequent local duplicate audit verified and preserved the archive, decoded all 410 samples, found no exact decoded-pixel groups, and investigated 23 candidate pairs. One pair has plausible shared-scene content; 22 match mainly through presentation/layout. None crosses the official train/test boundary, which does not establish subject/video independence. Two unparsed XMP blocks and missing authoritative source mappings remain UNVERIFIED. No identities, source-video IDs or project splits were created.

## Proposed research architecture

```mermaid
flowchart LR
    R[RGB video and timestamps] --> F[Face crop and HRNet landmarks]
    R --> S[SegFormer-B0 body parsing]
    R --> P[RGB infant pose adapter: future validation]
    R --> O[Optical flow]
    F --> Q[Aligned features and quality masks]
    S --> Q
    P --> Q
    O --> Q
    Q --> T[TCN movement prediction]
    D[BabyPose depth images] --> B[Independent depth pose experiment]
    R --> H[Future Hiera video comparison]
```

Facial landmarks describe geometry; expression recognition requires additional validated expression labels. Segmentation measures body regions and gates flow, while pose measures joints. These branches meet only when observations refer to the **same video and timestamps**. BabyPose depth supervision remains a separate experiment; it does not supply RGB labels.

## Independent experiments

| ID | Experiment | Supervision / output |
|---|---|---|
| E0 | Provenance, annotation, and split audit | Dataset-specific manifests and leakage report |
| E1 | HRNet facial landmark adaptation | InfAnFace landmarks; normalized localization errors |
| E2 | SegFormer-B0 body parsing | YouTube Infant Body Parsing masks; region overlap |
| E3 | Infant depth-pose adaptation | BabyPose joints; 2D localization errors |
| E4 | Movement features and causal TCN | Chambers pose sequences; nonclinical forecasting |
| E5 | Modular video integration | Synchronized permitted RGB videos; quality and robustness |
| E6 | Deferred Hiera video comparison | Same eligible forecasting target and held-out groups |

These datasets must never be concatenated into a single supervised dataset. E1–E4 have separate adapters, label schemas, configurations, checkpoints, and reports. E5/E6 are conditional on access and validated inputs.

## Repository layout

```text
README.md / AGENTS.md / research_plan.md
pyproject.toml    # Installable package, development tools, lint/test configuration
configs/
  datasets/       # One schema and configuration per dataset
  models/         # Architecture, source revision, checkpoint identity
  experiments/    # E0–E6 resolved protocols, created after review
  environments/   # CPU / Apple MPS / optional later CUDA overrides
src/infant_movement/
  datasets/       # Metadata envelope and independent InfAnFace annotation reader
  models/         # Face, segmentation, depth pose, RGB pose, TCN adapters
  features/       # Geometry, flow, timestamps, missingness
  engine/         # Training, checkpointing, inference, device selection
  evaluation/     # Grouped checks, whole-image candidates and local XMP references
  utils/          # Config, seeding, provenance, hashing
scripts/          # Explicit local-source metadata audit; training tools deferred
tests/            # Synthetic metadata, reader, image, provenance and CLI checks
manifests/        # Private actual sample, identity, and split records
data/{raw,interim,processed}/  # Local ignored assets, kept separate by dataset
artifacts/{checkpoints,features}/  # Local ignored derived artifacts
reports/          # Shareable aggregate reports after disclosure review
notebooks/        # Exploration using package code, not training logic
environments/     # Tested M1 development lock and environment/verification record
```

The original folder structure and `research_plan.md` are preserved. The plan records the original proposals and 18 October evidence snapshot; README and dated reports record implementation status. InfAnFace supervision stays separate from other datasets. The core package has no required runtime dependencies; image decoding uses only the optional hash-locked Pillow extra.

## Development environment

Verified for M1: **CPython 3.13.3, macOS 27.0.1, arm64**, with the exact tools in [the development lock](environments/macos-arm64-py313-dev.lock). See [the environment and verification record](environments/m1-environment.md). Python 3.11 remains the plan's proposed future training baseline; it was not installed for this standard-library-only milestone. Other Python versions, operating systems, and training/accelerator compatibility have not been verified. `requires-python >=3.11` is the code's intended minimum, not a tested compatibility matrix.

The existing `.venv` now also contains Pillow 13.0.0 for the bounded image audit. See [the optional decoder environment record](environments/infanface-image-audit.md) and [combined image-audit lock](environments/macos-arm64-py313-image-audit.lock). Historical M1 pins remain unchanged.

From the repository root, the existing environment is ready to use without activation:

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m pip --disable-pip-version-check --no-cache-dir check
.venv/bin/infant-movement audit tests/fixtures/synthetic_metadata.json
```

For a fresh checkout on the same platform with Python 3.13.3 already available, create the environment and install the hash-locked tools. The first installation command explicitly accesses the Python package index for packaging/testing tools only. It downloads wheels; it does not acquire research assets. The editable project installation then runs with network resolution disabled:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip --disable-pip-version-check --no-cache-dir install --only-binary=:all: --require-hashes -r environments/macos-arm64-py313-dev.lock
.venv/bin/python -m pip --disable-pip-version-check --no-cache-dir install --no-index --no-build-isolation --no-deps -e .
```

The lock includes pip, setuptools, wheel, pytest, Ruff, and all their installed dependencies. `pyproject.toml` also declares a pinned `dev` extra; use the lock above for the verified complete environment. Avoid installing an unpinned tool midway through a milestone. M1 needs no decoder, PyTorch, torchvision, notebook runtime or accelerator tooling. The optional image-audit extra adds only Pillow; use its combined lock for the full 150-test environment. Imports, audits, tests, and lint checks make no network requests. For offline recreation, obtain the locked wheels explicitly on an authorized connection, then add `--no-index --find-links /path/to/wheels` to the locked-tools install command. Hash checking follows [pip's documented installation mode](https://pip.pypa.io/en/stable/topics/secure-installs/); package configuration follows the [PyPA guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).

To verify a distributable wheel locally, after the checks above:

```sh
.venv/bin/python -m pip --disable-pip-version-check --no-cache-dir wheel --no-index --no-build-isolation --no-deps --wheel-dir dist .
```

`dist/`, `.venv/`, caches, and generated package metadata are ignored by Git. No license has been assigned to this project's code; dataset, original-media, and pretrained-weight permissions remain separate.

## InfAnFace local annotation reader

`infant_movement.datasets.infanface` provides `load_infanface_archive(path)` and `load_infanface_csv(path)`. Both read explicit local files and return immutable dataset-native samples. They perform no downloads, archive extraction, image decoding, split generation, or model inference.

The observed CSV contains 144 columns: four image/partition references, four binary attributes, and 68 `(x,y)` pairs. The reader checks finite coordinates, schema, reference safety, uniqueness, and official partition/attribute consistency. It preserves fractional and out-of-bounds source values. `landmark_bounds_mask(sample, width, height)` supplies a separate geometric QC mask; this is not a visibility annotation. The source pixel indexing origin remains UNVERIFIED.

For an aggregate local check of the already acquired archive:

```sh
.venv/bin/python - <<'PY'
from collections import Counter
from infant_movement.datasets.infanface import load_infanface_archive
samples = load_infanface_archive("data/raw/infanface/InfAnFace.zip")
print(len(samples), dict(Counter(s.official_split for s in samples)))
PY
```

Expected for the recorded receipt: 410 rows, with 210 `infanface-train` and 200 `infanface-test`. The archive SHA-256 and private evidence locations are recorded in the [acquisition audit](reports/infanface-e0-acquisition-audit-2026-10-27.md). An independent publisher checksum remains UNVERIFIED.

No subject/source-video IDs or project partitions are assigned. The private `manifests/infanface-unassigned-inventory-2026-10-27.json` intentionally does **not** satisfy the M1 split schema: unknown source groups and unassigned project partitions must not be fabricated to make an audit pass. Do not feed that inventory to the `audit` CLI as a claimed clean split. Every other dataset's supervision remains separate.

## Local duplicate and provenance audit

The completed [dated report](reports/infanface-duplicate-provenance-audit-2026-10-27.md) documents the fixed method and all aggregate findings. Exact equality uses dimensions plus decoded RGBA8 pixels, both stored and EXIF-oriented. Near-candidate screening uses whole-image dHash ≤6 **and** aHash ≤8 on the same view combination, testing full/central-80% views and mirrors. No face recognition, embeddings or infant identity assignment is used. Similarity is a review request, not duplicate/source confirmation.

All 83,845 pairs were compared. There were zero exact-pixel groups and 23 candidates, all within official training. All candidates were investigated; one remains a plausible shared-scene pair, while 22 are primarily layout/presentation matches. Source review found no confirmed video mapping; two XMP blocks remain unparsed/UNVERIFIED. These findings do not certify patient/video independence or justify a new independent split.

To recreate the optional decoder environment on this platform, explicitly install the hash-locked wheels and refresh the local editable package without dependency resolution:

```sh
.venv/bin/python -m pip --disable-pip-version-check --no-cache-dir install --only-binary=:all: --require-hashes -r environments/macos-arm64-py313-image-audit.lock
.venv/bin/python -m pip --disable-pip-version-check --no-cache-dir install --no-index --no-build-isolation --no-deps -e .
```

From the repository root, choose **new, unused** private output names:

```sh
.venv/bin/python -m infant_movement duplicate-audit --archive data/raw/infanface/InfAnFace.zip --receipt data/raw/infanface/acquisition-receipt-2026-10-27.json --output reports/private/infanface-duplicates-rerun.json
.venv/bin/python scripts/audit_infanface_sources.py --archive data/raw/infanface/InfAnFace.zip --audit-report reports/private/infanface-duplicates-rerun.json --output reports/private/infanface-sources-rerun.json
```

These commands read local files, verify archive integrity before/after, print aggregates and write per-sample evidence only under `reports/private`. They refuse overwriting and make no network requests or project assignments. Source-audit completion does not imply every XMP block parsed; inspect `xml_failure_count`. Original private audit JSON, contact sheets and review decisions remain Git ignored. Do not publish per-sample hashes, metadata or candidate edges. No images or data-dependent fixtures are embedded in tests.

## Metadata and grouped split audit

The implemented command reads a **local JSON metadata file only**, validates it, and checks already assigned project partitions. It neither generates splits nor reads images, checks actual annotation payloads, changes input files, or assigns clinical labels. Run it **before** frame extraction, windows, augmentation, learned preprocessing, or pseudo-labeling. Audit each CV fold/protocol separately; a manifest has exactly one row per `(dataset_id, release_id, sample_id)`.

Use [the entirely synthetic example](tests/fixtures/synthetic_metadata.json) as the format reference:

```json
{
  "schema_version": 1,
  "records": [
    {
      "dataset_id": "synthetic",
      "release_id": "v1",
      "sample_id": "sample-001",
      "source_group_id": "source-001",
      "project_split": "train",
      "identity_status": "unknown",
      "subject_id": null,
      "source_video_id": "video-001"
    }
  ]
}
```

| Field | M1 validation rule |
|---|---|
| `dataset_id`, `release_id`, `sample_id` | Required pseudonymous identifiers; the tuple uniquely identifies one row |
| `source_group_id` | Required conservative source/provenance group, including still images; a proxy does not establish infant identity |
| `project_split` | Required, exactly `train`, `validation`, or `test`; partial manifests are permitted and counts include absent partitions |
| `identity_status` | Required, exactly `known` or `unknown` |
| `subject_id`, `identity_evidence` | Nullable; `known` requires both a shared subject key and a private evidence-reference key; `unknown` requires `subject_id=null` |
| `source_video_id` | Nullable for still images/unresolved sources; when known, all derivatives inherit the same canonical registry key |
| `duplicate_group_id` | Nullable key linking known exact/near duplicates and reuploads, including different video IDs |
| `session_id` | Optional nullable reference metadata; it is not a substitute for source/subject grouping |
| `official_split` | Optional nullable original split name; retained exactly, never used to overwrite project assignments |
| `frame_index` | Optional nonnegative integer; booleans/fractions rejected |
| `timestamp_seconds` | Optional finite nonnegative number; neither timing nor a frame rate is inferred |

Identifiers use 1–128 ASCII letters/digits plus `_`, `.`, `:`, `-`, starting with a letter/digit. Empty strings, whitespace, URLs, and placeholder strings such as `unknown` or `UNVERIFIED` are rejected as identifiers. `official_split` is a nonempty original name and may contain spaces or punctuation. Use JSON `null` for missing optional values; required unknown provenance must be resolved or excluded before this audit can pass. Omitted optional fields become `null`, never a negative label or zero measurement. Frame/time fields require a source-video key. Unsupported fields, duplicate JSON keys, nonfinite numbers, invalid identity combinations, empty manifests, and unsupported schema versions fail validation. This schema checks single-row timestamp validity; sequence ordering and timestamp/media alignment remain unimplemented.

`subject_id`, `source_video_id`, `source_group_id`, and `duplicate_group_id` are **global keys from a common private registry**, with a separate namespace for each field. Do not prefix a shared video's key with a dataset name or change it between releases. Resolve dataset-local aliases and known reuploads in that private registry first. For unrelated local IDs, assign distinct pseudonymous registry keys. The grouped-key checker performs no facial identification or duplicate discovery; the separate optional whole-image audit generates candidates for evidence review, never infant identities. A registry can include metadata references across collections to detect overlap; their labels and training datasets remain separate.

Each supplied subject, source video, source group, and duplicate group must occur in at most one project partition. Checking every relationship also catches a connected group crossing partitions: at least one shared-key relationship must cross the boundary. Repeated sample keys fail even within one partition; multiple frames from one source inside a partition are valid. Null subjects/videos are skipped and counted as limitations. A clean result does **not** establish patient independence, rights clearance, absence of unidentified duplicates, or upstream-training disjointness.

To audit an eventual approved private manifest:

```sh
mkdir -p reports/private
.venv/bin/infant-movement audit manifests/private-metadata.json > reports/private/split-audit.json
```

The input above is a future private file, not a provided dataset. `python -m infant_movement audit ...` is equivalent when using the environment's Python. Exit codes are **0** for valid metadata with no known group overlap (warnings may remain), **1** for repeated samples or group leakage, and **2** for invalid/unreadable input or command usage. JSON output includes counts, warning/limitation text, and one-based record rows for conflicts. It omits registry IDs. Actual manifests, identity evidence/maps, and audit reports must remain private; the ignore rules cover `manifests/` and `reports/private/`.

## Verification and next review boundary

Current verification: **150 synthetic tests passed**, with no skips in the image-audit environment (the previous 121 plus 29 image/provenance checks); Ruff lint/format and dependency consistency passed. All 410 real images decoded, and raw checksums remained unchanged. Synthetic tests require no infant data. See [the duplicate audit](reports/infanface-duplicate-provenance-audit-2026-10-27.md) and [decoder record](environments/infanface-image-audit.md). The historical M1 and acquisition reports are unchanged. No model/accelerator or scientific generalization result is claimed.

**Stop after this bounded duplicate/provenance audit.** The next proposed step is independent adjudication of private scene/source clues and resolution of a documented grouping basis, plus an explicit coordinate/orientation policy before E1 preprocessing. Preserve official partitions as metadata; no patient/video-independent validation split can yet be certified. No project splits, weights, models, training, inference, further datasets/media, contact, uploads or remote publication are part of this phase.

## Review focus

Review the M1 metadata contract alongside the dataset access uncertainties, conservative split protocols, forecasting target, and deferred RGB-pose validation in [research_plan.md](research_plan.md). The synthetic checks establish no training reproducibility or clinical usefulness; both require subsequent evidence.
