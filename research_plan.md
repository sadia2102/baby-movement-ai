# Research plan: Infant Movement Analysis Using Deep Learning

**Status:** proposed protocols, no implementation or results. **Evidence checked:** 18 October 2026. **Scope:** independent biomedical computer vision research, not medical diagnosis.

## 1. Research questions and permitted conclusions

The immediate goal is a reproducible measurement pipeline: locate facial landmarks, segment body regions, estimate joints in the appropriate image modality, and characterize their dynamics. The first temporal objective is prediction of future movement features, which can be evaluated without clinical event labels.

The long-term seizure-related question remains a separate, unvalidated hypothesis. None of the four candidate resources is verified here as a seizure-annotated dataset. A source calling infants healthy does not establish absence of seizures, and neuromotor risk does not establish their presence. Forecast error or statistical unusualness must never be renamed seizure probability.

Questions for this phase of the research:

1. Does infant-specific facial adaptation improve localization over a frozen adult-pretrained baseline?
2. Does a small pretrained segmentation model reliably separate infant body regions under occlusion and changing appearance?
3. How accurately can a compact infant-adapted model localize 2D joints in depth images of unseen infants?
4. Do measured movement features and a causal TCN improve forecasting over persistence and linear baselines on held-out recordings?
5. How do missing detections, camera movement, and upstream model errors change temporal measurements?

No clinical utility, emotion understanding, seizure detection, or RGB generalization claim follows automatically from answering these questions.

## 2. Dataset evidence and access register

**VERIFIED** below means supported by the linked public primary source, not inspected inside downloaded archives. **UNVERIFIED** means evidence was unavailable, incomplete, or requires archive inspection. No dataset archive, frame, annotation file, or model weight was fetched during planning. Counts are source-reported and must be reconciled against a pinned release later.

### 2.1 InfAnFace — E1

- **URLs:** [official repository](https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking), [author-hosted dataset directory](https://coe.northeastern.edu/Research/AClab/InfAnFace/), [paper](https://arxiv.org/abs/2110.08935).
- **VERIFIED:** 410 RGB face images; 68 Multi-PIE landmarks plus turned, occluded, expressive, and tilted attributes in CSV. Suggested train: 210 (`ads`, `google2`, `youtube2`); test: 200 (`google`, `youtube`), including 80 Common and 120 Challenging. Collection batches define the split, not certified infant identities. No separate validation partition is documented in the dataset description.
- **License:** dataset terms restrict use to noncommercial purposes and exclude diagnostic/product uses; the repository separately identifies MIT code licensing. Do not apply the code license to the images. See the repository's dataset terms.
- **UNVERIFIED:** unique subject IDs, source-video grouping completeness, cross-batch duplicates, archive integrity, and current checkpoint rights.
- **Boundary:** isolated images support landmark adaptation. The expressive attribute is not a temporal expression taxonomy or seizure label.

### 2.2 YouTube Infant Body Parsing — E2

- **URL:** [official repository and frame-access link](https://github.com/nihaomiao/Youtube-Infant-Body-Parsing).
- **VERIFIED:** 90 RGB videos from the Chambers collection; sampled every four frames. Five semantic classes: background, head, arm, torso, leg. Official video split: 68 train / 22 test. Train: 2,149 labeled + 4,690 unlabeled frames; test: 1,256 labeled + 2,737 unlabeled. Four JSON files encode these partitions; labels are base64 RGB masks. No validation split is listed.
- **License:** the authors state research-only release; a standard dataset license and rights to redistribute underlying YouTube media are **UNVERIFIED**. The repository links sampled frames on Google Drive; access and contents were not tested.
- **UNVERIFIED:** subject-disjointness, original-to-local video mapping completeness, precise timestamps, and available source videos.
- **Boundary:** arm/leg classes do not distinguish sides. Unlabeled frames are not background-only examples. Overlap with Chambers must be audited before downstream evaluation.

### 2.3 BabyPose — E3

- **URLs:** [original release v1](https://zenodo.org/records/3891404), [current linked v2](https://zenodo.org/records/5336836), [data paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7551984/).
- **VERIFIED in the data paper:** 16 preterm infants, 16,000 depth frames, 480 × 640, 8-bit and 16-bit PNG representations. Twelve 2D joints cover bilateral shoulders, elbows, wrists, hips, knees, and ankles. Negative coordinate pairs mark occlusion. Original XLSX split: 750 train + 250 test frames **per infant**, totaling 12,000 / 4,000. No validation set is described.
- **Version caveat:** v2 lists a changed archive and annotation description; its internal schema and correspondence to the paper are **UNVERIFIED**. Do not silently mix releases.
- **License:** dataset license **UNVERIFIED**; the rendered Zenodo rights field did not expose a license value. The article's CC BY-NC-ND notice is an article license, not proof of dataset permissions. Resolve dataset terms before use.
- **Boundary:** depth measurements with 2D keypoints do not constitute annotated 3D joints or paired RGB supervision.

### 2.4 Chambers Infant Movement Assessment — E4

- **URLs:** [official code](https://github.com/cchamber/Infant_movement_assessment), [public Figshare DOI](https://doi.org/10.6084/m9.figshare.8161430), [author-linked Figshare share](https://figshare.com/s/10034c230ad9b2b2a6a4), [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8011647/).
- **VERIFIED:** Figshare lists YouTube and laboratory metadata, pose-estimate CSVs, video URL lists, and supporting assets; the public record reports **CC BY 4.0**. That record license does not establish rights to original videos. Its version history differs from the author-linked private-share view; pin the exact public version and reconcile contents later.
- **Annotations:** released trajectories are model-estimated poses, not automatically manual ground truth. The code describes normative movement and laboratory neuromotor-risk comparisons and says ground-truth pose labels are not provided there. [Repository evidence](https://github.com/cchamber/Infant_movement_assessment).
- **Splits:** a reusable, subject-disjoint train/validation/test manifest for temporal learning is **UNVERIFIED**. Do not treat the paper's reference/clinical cohorts as such a split.
- **UNVERIFIED:** current raw-video availability, timestamps/FPS, complete identity mappings, per-file licenses, risk-field definitions, and upstream pose-training overlap.

### 2.5 Evidence gaps to resolve after review

| Resource | Required next evidence | Effect until resolved |
|---|---|---|
| InfAnFace | Archive schema, grouping evidence, duplicate audit, checkpoint training provenance | E1 remains conditional; no certified patient-independence claim |
| Body parsing | Research terms, frame access, palette integrity, source-video and subject mappings | No acquisition/use before terms are resolved; no patient-disjoint assertion |
| BabyPose | Dataset license, chosen release, depth scale/invalid values, annotation changes | E3 acquisition and preprocessing are gated |
| Chambers | Version/file inventory, timestamp and identity fields, pose-estimator training membership | E4 may be limited to video-disjoint pseudo-target forecasting |
| Cross-dataset | Shared videos, infant/session IDs where supplied, reuploads and source batches | E5 cannot use an upstream branch trained on downstream held-out recordings |

Record URL, access date, version/DOI, terms, source revision, annotation provenance, checksum, and exclusions in a private dataset card at acquisition. These resources are not one cohort; do not sum their counts as independent infants. Public availability alone does not establish permission for redistribution or new clinical use.

## 3. Architecture and interfaces

Use independent PyTorch modules and ordinary training loops. This keeps modality assumptions visible, allows each branch to fail independently, and makes synthetic correctness tests possible without loading every model. Avoid an end-to-end multimodal network until the single-task experiments and aligned inputs are validated.

| Component | Proposed contract | Architectural reason and weakness |
|---|---|---|
| Face localization + HRNet | RGB face crop → 68 heatmaps, original-image coordinates, confidence, crop transform | High-resolution features suit localization; face detection/cropping is an additional dependency and may dominate failures |
| SegFormer-B0 | RGB frame → five-class probabilities and mask in original coordinates | Compact segmentation branch; adult scene pretraining does not encode infant body-part semantics |
| Depth pose | Depth tensor + invalid-depth mask → 12 heatmaps, pixel coordinates and validity | Matches BabyPose modality; cannot be substituted directly for RGB pose |
| RGB pose, conditional | RGB frame → declared skeleton, per-joint confidence and missingness | Needs a separate infant RGB checkpoint and independent validation; no current dataset supplies the complete paired pipeline labels |
| Movement features | Timestamped observations → feature matrix, feature masks, provenance | Makes geometry and uncertainty auditable; noisy keypoints can manufacture motion |
| Causal TCN | Past features and masks `[B,F,T]` → fixed-horizon movement forecasts | Small temporal model for cached inputs; receptive field and sampling rate limit measurable dynamics |
| Hiera, deferred | RGB clip → video embedding → same declared forecast target | Tests video representation value; more memory, source-domain bias, and weaker interpretability |

The InfAnFace repository supplies HRNet-based facial models but documents a legacy runtime. Port only the required network/decoding adapter after checking licenses and coordinate parity; do not recreate its obsolete environment on the Mac. Treat a detector-backed pipeline and ground-truth-crop landmark evaluation as separate protocols. [Official model implementation](https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking).

For segmentation, a candidate initialization is `nvidia/segformer-b0-finetuned-ade-512-512`; replace the incompatible classification head with five channels and log newly initialized parameters. Keep background as a supervised class, set `do_reduce_labels=False`, and reserve a different value for genuinely ignored pixels. Resize masks with nearest-neighbor interpolation and logits to evaluation resolution with a declared bilinear setting. [Processor documentation](https://huggingface.co/docs/transformers/model_doc/segformer), [checkpoint card](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512).

The NVIDIA checkpoint card points to the [SegFormer source license](https://raw.githubusercontent.com/NVlabs/SegFormer/master/LICENSE), which limits covered work to noncommercial research/evaluation. Pin and review the exact artifact terms. Hugging Face library licensing does not replace pretrained-weight terms. Hiera likewise separates Apache-2.0 code and CC BY-NC 4.0 model weights. [Hiera source and licensing](https://github.com/facebookresearch/hiera).

### 3.1 Planned sample and artifact schemas

Use a common metadata envelope, while keeping task-specific payloads and readers separate:

```text
Sample metadata:
  dataset_id, release_id, sample_id, modality, relative_path, sha256
  subject_id (nullable), session_id (nullable), source_video_id (nullable)
  source_group_id, duplicate_group_id, identity_status, identity_evidence
  frame_index, timestamp_seconds (nullable), fps_source, width, height
  official_split, project_split, fold, annotation_kind, annotation_version
Task payload: landmarks + visibility OR class mask OR joints + validity
```

Missing identity is null, not a fabricated patient ID. `source_group_id` supplies the conservative auditable grouping unit when infant identity is unavailable. Mark it as a proxy. A separate relationship table links overlapping releases/datasets without mixing their labels. Coordinate arrays always declare `(x,y)` order, pixel-center convention, units, skeleton layout, and the invertible transform from model to source coordinates.

Feature artifacts additionally store source timestamps, feature names/units, masks, upstream model and training-split hashes, crop/resize settings, preprocessing revision, and raw-vs-predicted annotation provenance. Use local NPZ initially; add Parquet only if scale warrants another dependency. Never zero-fill missing joints without an explicit mask. Cache keys must include upstream training membership, not merely model architecture.

Configuration will declare experiment ID, one dataset/release, split file/hash, source/weight revisions, target, input geometry, augmentation, optimizer/scheduler, batch and accumulation, seed, device/dtype, metric definitions, and output path. Future entry points will be explicit audit, train, extract-features, and evaluate commands; none exists yet.

## 4. Split and validation strategy

### 4.1 Rules applying to every experiment

1. Establish infant → session → source-video → frame/window relationships before splitting. Reuploads and near-duplicate recordings form connected groups; no connected group crosses partitions. Do not automatically identify infants by faces. Use supplied metadata and documented source review; quarantine ambiguous overlaps when necessary.
2. Split at the highest reliable unit: infant/patient first, otherwise conservative source group. All frames, crops, depth encodings, augmentations, unlabeled samples, and windows inherit the parent partition. For unknown identities, video-disjointness is the strongest supportable claim, and repeated infants may remain a residual limitation.
3. Preserve official membership separately. Do not repair it silently. A limited official reproduction may be reported in a separate table; the clean project protocol is primary. If identities conflict with an official holdout, do not use overlapping training examples; document exclusions and loss of benchmark comparability.
4. Fit normalization, imputation, PCA, feature selection, learned filters, pseudo-labeling teachers, thresholds, and sampling policies on training groups only. Tune hyperparameters and checkpoints on validation; keep final test untouched. Per-clip deterministic geometry normalization may use observed input frames, never future target frames.
5. Divide recordings before generating temporal windows. Keep input context, receptive-field context, and prediction targets inside one recording and partition. No windows bridge cuts, missing-time gaps, or sessions. For any optional within-recording diagnostic split, purge both input context and forecast horizon and label the result within-subject only.
6. A downstream test recording must not have contributed to upstream fine-tuning, self-supervision, or pseudo-label teacher adaptation. Train upstream models within each temporal outer fold or freeze a model with documented disjoint provenance. Unknown pretrained contamination is disclosed, not assumed away.
7. Test data cannot be used as unlabeled training data. Exclusion rules and quality thresholds are fixed using training/validation evidence; report test rejection counts and coverage so abstention cannot hide failures.

Grouped folds can use `GroupKFold`/`GroupShuffleSplit`; these tools enforce only the groups supplied and cannot repair missing identities. [scikit-learn grouped validation documentation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).

### 4.2 Proposed concrete protocols

| Experiment | Training and validation | Final evaluation |
|---|---|---|
| E1 | Retain official training pool; form a deterministic approximately 80/20 group split after audit. No random frame/image split when a source relationship exists | Lock official test if group-clean. Otherwise quarantine conflicts and report the clean subset as a changed protocol; unknown identity precludes patient-generalization claims |
| E2 | Keep official test videos sealed; approximately 80/20 group split within official training videos. Use patient groups where known | Official held-out videos after source and identity conflict audit; labeled masks only for segmentation scoring |
| E3 | Primary: nested four-fold outer patient CV over all eligible infants; with the documented 16 subjects this gives four outer test infants per fold. Use three-fold patient CV inside the remaining 12 for tuning | Each patient is outer-test exactly once. Choose epoch count from inner validation, then refit on outer-training patients; report pooled out-of-fold results. The published overlapping split is optional, separately labeled within-patient reproduction |
| E4 | Within Chambers normative source groups, reserve roughly 20% as sealed test with seed 42; use three-fold grouped CV on the remainder for a small fixed search. Keep laboratory records separate | One final evaluation on sealed normative groups; laboratory data may be descriptive external analysis only after schema and identity review, without clinical label reinterpretation |
| E5/E6 | Inherit the same eligible E4 groups and holdouts; enforce branch-training exclusions across collections | Common held-out recording subset for paired comparisons; report attrition and the full cohort separately |

Proportions are group targets, not promises of exact frame counts. Freeze actual IDs and counts after E0. If group counts cannot support the proposed validation folds, revise the protocol before training, publish the reason, and preserve a final holdout. Do not optimize the seed or partition for favorable results.

### 4.3 Repeated runs and statistical unit

Use training seeds 42, 43, and 44 for final selected settings when resources permit; label a one-seed pilot as such. Fix partitions independently of model seeds. Report mean and standard deviation across seeds; they are not confidence intervals over patients.

Compute primary metrics per infant and average infants equally. If identity is absent, average per video/source group and state the limitation. Bootstrap 95% intervals by resampling independent groups, not frames, using 2,000 resamples and a recorded bootstrap seed. For comparisons, bootstrap paired per-group metric differences. With very few patients, show every patient result and interpret intervals cautiously. Cross-validation folds are not independent patients.

## 5. Independent experiment specifications

All hyperparameters here are initial proposals, not tested settings. Each experiment produces its own data audit, frozen protocol, resolved config, checkpoint provenance, validation curves, held-out metrics, and failure analysis. E1, E2, E3, and E4 can progress independently after E0; failure to acquire one dataset must not trigger label borrowing from another.

### E0 — Dataset and reproducibility audit

**Purpose:** make evidence gaps explicit before modeling. After authorization, inventory one approved resource at a time, hash assets, validate annotation ranges and dimensions, inspect subject/source mappings, and preserve official split lists. Do not obtain all datasets merely because one experiment is ready.

Check exact and near duplicates across candidate train/test pools and shared video sources. Save a private provenance registry and an aggregate overlap report. Validate RGB palette masks, negative joint handling, the pairing of BabyPose encodings, timestamps, and model-estimated versus manual annotations. Keep canonical URLs private where they identify children.

**Acceptance:** every eligible sample has a release and grouping basis, no known connected group intersects partitions, all exclusions have reasons, and unresolved rights block that resource. Acceptance includes explicit limits where patient identity cannot be established; it does not certify patient independence from missing metadata.

### E1 — Facial landmark localization

**Target:** landmark coordinates, not seizure status or emotion classes. Compare adult-pretrained HRNet without infant tuning, a frozen-backbone/retrained-head baseline, then infant fine-tuning. A published infant checkpoint can be a reference only when its training membership and preprocessing are known; it cannot count as an untouched baseline for a new split it may already have seen.

Start at 256 × 256 crops, batch 2, masked heatmap MSE, AdamW, head LR 1e-3 and backbone LR 1e-4 as candidates, with weight decay 1e-4. Use up to 60 epochs, patience 10 on validation NME, and select an epoch count using validation only. A small predeclared search may compare backbone LR 1e-5 versus 1e-4. Freeze batch-normalization statistics at small batch sizes; compare staged unfreezing. Rotation, scale, and horizontal flips must transform landmarks and swap left/right indices. Do not import adult-data joint-training recipes without a separately authorized dataset and license review.

**Primary metric:** NME = mean landmark Euclidean error divided by ground-truth interocular distance, macro-averaged by group. Match the source implementation's exact eye indices/definition before claiming benchmark comparability. Predeclare a documented fallback bounding-box-diagonal NME for invalid/degenerate eye distances and report it separately; do not silently replace the primary denominator. **Secondary:** NME CED-AUC to 0.10, failure rate above 0.10, per-landmark error, Common/Challenging and pose-attribute breakdowns. Thresholds are research conventions, not clinical cutoffs.

Score ground-truth-crop localization separately from detector-backed performance. Count missed faces, crop failures, and valid-frame coverage in the latter. A non-detection cannot disappear from the report. **Weakness:** a small still-image set cannot establish temporal tracking, expression recognition, or robustness in clinical video.

### E2 — Five-class body parsing

**Target:** dataset-defined masks. First use labeled training frames only. Compare a compact randomly initialized U-Net baseline, SegFormer-B0 head-only adaptation, and gradual encoder fine-tuning. Keep a fixed label map: background 0, head 1, arm 2, torso 3, leg 4. The repository's decoder uses alternative variable names; validate meanings against the annotation description rather than renaming anatomy from variable names.

Start with aspect-preserving 256-pixel inputs and padding; train with masked cross-entropy plus foreground soft Dice. Use batch 1–2 and accumulation to effective batch 8, frozen-backbone warmup 5 epochs, then at most 50 total epochs. Candidate AdamW head/backbone rates: 1e-3 / 1e-4, with a 1e-5 backbone alternative; select by validation foreground mIoU, patience 10. Class weights, if used, come only from training masks. Apply consistent image/mask transforms; unknown palette values cause audit failure, not automatic background conversion.

**Primary:** per-video mean IoU averaged across four foreground classes, then macro-averaged over groups; also report five-class mIoU including background. **Secondary:** per-class Dice/IoU, boundary F1 with a declared original-resolution tolerance (initial proposal: 2 pixels), confusion matrix, and coverage. For class-absent frames, omit undefined unions and disclose support; a false-positive region gives a nonzero union and must be penalized. Report a pooled pixel confusion-matrix metric separately for comparability.

Only after the supervised baseline, consider a separate semi-supervised arm using **training** unlabeled frames, a training-only teacher, and validation-chosen pseudo-label confidence. No label propagation across partitions. **Weakness:** merged left/right limbs, boundary ambiguity, source-video bias, and missing patient identity constrain conclusions.

### E3 — Infant depth pose estimation

**Target:** pixel locations for the dataset skeleton. Begin with a small heatmap CNN trained from scratch, then compare adaptation from a declared pretrained pose backbone. For an RGB-derived backbone compare three-channel depth replication with a one-channel stem initialized by summing RGB input-channel weights; summation matches the initial response to replicated depth under the same normalization. Log normalization differences and reinitialize the task head. This is a modality-adaptation experiment, not an established infant RGB model.

Preserve original 16-bit data and an invalid-depth mask. Sensor units, scale, and missing-value codes are acquisition gates; do not assume every stored value is millimeters. Compare 8-bit and 16-bit inputs as separate runs on identical patient folds, never as independent samples. Fit depth clipping/normalization bounds independently on each inner-training fold during tuning, and on outer-training patients only for the final refit. Mask occluded/negative joints from supervised loss and preserve missingness in reports.

Propose aspect-preserving 256-pixel inputs, masked heatmap MSE, AdamW LR 1e-3 from scratch or 1e-4 for adaptation, 60 epochs maximum, patience 10 on inner-validation normalized joint error. Fix this small search before nested CV.

**Primary:** patient-macro PCK@0.10 of ground-truth torso length, where torso length is the distance between shoulder midpoint and hip midpoint. **Secondary:** PCK@0.05 and 0.20, PCK-AUC over [0,0.20], normalized joint error, pixel error at native resolution, per-joint/support results, and visibility-stratified coverage. Torso metrics require valid shoulders/hips; report excluded cases and pixel error on other eligible joints. Do not derive torso normalization from predictions or call pixel errors 3D MPJPE. **Weakness:** small neonatal sample, depth-specific artifacts, and unavailable occluded-joint truth; low pose error does not validate movement pathology.

### E4 — Movement features and a causal TCN

**Feasible first target:** forecast future pose-derived wrist displacement on Chambers normative trajectories. This uses trajectory pseudo-targets and provides no independent proof of anatomical truth. Do not train a classifier using normative versus clinical source as a disease label. If upstream pose-training membership cannot be established, report possible transductive contamination and limit the result to feasibility.

First extract quality-aware features: trunk-normalized joint displacement, joint angles, speed, acceleration, bilateral motion correlation, and missing-joint fraction. Keep global translation and body-relative articulation as separate features. Derivatives use actual delta-time; no spectral claims without verified timing. Compute rotation and scale from valid **past input** poses; retain a fixed transform for each forecast target. Use causal short-gap interpolation only (initial maximum 0.2 s), masks, and no bridging long gaps. Smoothing must not access future targets. Run unsmoothed and smoothed ablations because smoothing can erase rapid movements.

When timing permits, resample to 10 Hz with documented anti-aliasing and gaps; otherwise forecast in frame steps and do not claim real-time rates. Proposed input context: 2 seconds; horizons: 0.5 s primary and 1 s secondary. Target the 2D displacement of each valid wrist between the last input frame and the horizon in the input-defined trunk-normalized coordinate frame. Context and target quality requirements are fixed on development data.

Compare zero-displacement persistence, constant-velocity extrapolation, ridge regression over the same context, then a causal residual TCN. Initial TCN: three blocks, two convolutions per block, kernel 3, dilations 1/2/4, 32 channels, dropout 0.1. Its receptive field is 29 frames; with 20 input samples, document left-padding and that effective real history is 20 samples. Use masked Huber loss, AdamW LR 1e-3, batch 16 feature windows, at most 100 epochs, patience 10 on validation endpoint error. Limit the initial search to widths 32/64 and LR 1e-3/1e-4; no tuning on the sealed test.

**Primary:** mean Euclidean wrist displacement endpoint error in trunk units at 0.5 s, per video then group-macro. **Secondary:** coordinate MAE/RMSE, 1 s error, per-wrist error, relative error reduction over persistence (undefined if baseline error is zero), eligible-window coverage, and robustness to synthetic dropout/noise. Preserve contiguous time blocks for diagnostics; do not treat overlapping windows as independent observations. No AUROC or seizure sensitivity is meaningful for this target.

Optional descriptive unusualness scores must use a training-only reference distribution and validation-defined thresholds. They remain movement-deviation scores with no seizure interpretation. Laboratory analysis requires a separate protocol accounting for age, acquisition environment, and source-domain confounding.

### E5 — Synchronized RGB integration and optical-flow ablations

**Gate:** permitted RGB video access, verified timestamps, and a validated RGB infant-pose branch. E3 cannot satisfy that RGB gate. Until then, integration can be designed and tested synthetically, but complete multimodal accuracy is unevaluable.

On the same recording, align face geometry, body masks, pose, and flow by source timestamps. Start with CPU Farneback flow for low setup cost, store vectors and quality information, and quantify camera motion using background regions when available. Calculate per-body-region flow magnitude/direction summaries; normalize by elapsed time and declared image/body scale. Keep RGB and depth flow as separate protocols. Without flow ground truth, do not report endpoint accuracy on real videos; synthetic translated images can validate basic vector direction, magnitude, and resizing.

Compare unmasked flow, predicted-mask flow, and annotation-mask flow on eligible labeled frames. For forecasting, use pose-only, flow-only, face-only where present, and combined features on a common subset; report full-cohort coverage separately. Use ground-truth versus predicted inputs only where paired annotations actually exist. Do not fabricate a multimodal gold-standard set from unrelated subjects.

**Metrics:** the E4 forecasting metrics when a valid common target exists; face/joint detection coverage, valid region fraction, gaps, temporal alignment errors, synthetic-flow endpoint error, and per-stage/end-to-end latency. Jitter or temporal consistency alone is not accuracy: a frozen prediction can score well on smoothness. Record synchronized timing, peak process/unified memory where measurable, and MPS synchronization for latency measurements.

**Weakness:** each branch can fail on the same occlusion, estimated masks can suppress real movement, face crop jitter can mimic expression change, and camera/caregiver motion can dominate flow. Report failure propagation and missing branches rather than silently imputing complete observations.

### E6 — Deferred Hiera video comparison

The official Hiera repository provides video variants distinct from image variants; a candidate is `hiera_base_16x224`, subject to licensing and runtime review. Use a genuine video checkpoint, not an image backbone described as a video transformer. [Official video model documentation](https://github.com/facebookresearch/hiera#video-models).

Freeze the video encoder initially and train a small regressor for the same E4 displacement target on the same accessible recordings and time horizons. Sample clips only before the forecast boundary; declare the actual time coverage of 16 frames. Compare pooled frozen video features plus a linear head with video features plus a TCN; separately compare the pose-feature TCN as a different-input system. Only matched-input comparisons can isolate the temporal architecture's effect.

Use identical outer groups, target generation, paired eligible recordings, validation search budget, and metric aggregation. Report checkpoint pretraining data, input duration, trainable/total parameters, latency, memory, and target-quality limitations. A different input modality, duration, or extra pretraining prevents attributing all improvement to Hiera. Full fine-tuning and large sweeps are deferred to suitable compute and new authorization. This experiment has no seizure-classification target.

## 6. Recommended dependencies and version policy

No dependency has been installed or compatibility-tested for this project. Use native arm64 wheels and a small dependency surface. Prefer Python 3.11 for the initial isolated environment; the workspace currently exposes arm64 Python 3.13.3, which was only inspected and has not been changed. Do not downgrade the user's system Python.

| Group | Recommended libraries | Purpose / necessity |
|---|---|---|
| Core training | `torch`, matching `torchvision` | Tensors, models, transforms; select the matched stable pair from official distribution guidance |
| Numeric and tabular | `numpy`, `scipy`, `pandas` | Arrays, filters/statistics, metadata; add only as each adapter needs them |
| Images and baseline flow | `Pillow`, `opencv-python-headless` | Image I/O, geometry, Farneback; install only one OpenCV distribution |
| Configuration | `PyYAML`; standard-library `dataclasses`, `argparse`, `json`, `pathlib`, `hashlib`, `logging` | Explicit configs and local provenance without a large training framework |
| Evaluation | `scikit-learn`, `matplotlib` | Grouped splits, ridge baseline, aggregate plots; task metrics can use tested NumPy/PyTorch functions |
| BabyPose only | `openpyxl` | Original XLSX annotations, conditional on chosen release schema |
| SegFormer only | `transformers`, `safetensors` | Model/processor adapter and safer weight serialization; pin tested APIs and disable implicit network access |
| Pose adapter, conditional | `timm` only if selected implementation requires it | Backbone dependency; infant-specific checkpoint not yet selected |
| Video I/O, conditional | `av`; optional system FFmpeg after need is established | Timestamp-aware decoding; verify target-format support on arm64 first |
| Hiera only, deferred | `hiera-transformer` and its tested requirements | Genuine video encoder; separate extra/environment, not core setup |
| Development | `pytest`, `ruff`; optional `ipykernel` | Synthetic correctness tests, linting, notebooks |

Avoid TensorFlow/Keras, OpenMMLab's compiled operator stack, CUDA toolkits, `flash-attn`, distributed launchers, large optical-flow models, and hosted tracking services in the initial environment. Ordinary PyTorch can express the intended CNNs and TCN. Do not add `torchaudio` when there is no audio experiment. `torchmetrics`, Hydra, Lightning, and MLflow can be reconsidered for a concrete need, but are not necessary to begin.

**Locking after review:** create a minimal `pyproject.toml` with task extras, resolve a compatible stable torch/torchvision pair plus direct/transitive dependencies in a clean arm64 Python 3.11 virtual environment, then commit exact versions and hashes in `environments/macos-arm64-py311.lock`. Record the patch version, macOS version, wheel platform, and resolver version. Keep a separate later Linux/CUDA lock; never assume binary parity. Run import/device checks and synthetic numerical tests before labeling a lock validated. Do not publish speculative pins as tested or install an arbitrary latest release midway through an experiment. [PyTorch installation guidance](https://pytorch.org/get-started/locally/).

## 7. Apple Silicon development profile

Use VS Code and a local `.venv` from a separate native arm64 Python installation after approval. A virtual environment is sufficient; Docker adds complexity for initial local accelerator work. Store datasets on local encrypted storage with enough space for originals and controlled caches. Hardware memory has not been inventoried; all settings below are conservative starting proposals, not measured capacity claims.

Start CPU-only with tiny synthetic fixtures. Later select MPS only when `torch.backends.mps.is_built()` and `is_available()` succeed; otherwise choose CPU explicitly. These checks and `torch.device('mps')` follow the [PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html). Confirm operation support with the exact locked build, not the existence of an Apple GPU alone.

- Float32 first; no automatic mixed precision or `torch.compile` until correctness is established.
- Image inputs around 256 pixels, batch 1–2, accumulation where useful, `num_workers=0`, `pin_memory=False`; benchmark 1–2 workers only later.
- Decode selected clips and cache compact features. Do not hold every frame or multiple image models in unified memory together. Run branches sequentially and reuse caches.
- Use CPU for unsupported operations or an explicitly separate CPU run. If enabling MPS CPU fallback for a prototype, record it and exclude hidden transfers from unsupported speed claims.
- Keep reproducibility reference tests on CPU. Set Python/NumPy/PyTorch seeds, loader generators, deterministic settings where supported, and log exceptions. Identical seeds do not guarantee bitwise equality between CPU, MPS, and CUDA.
- Freeze encoders for early TCN and Hiera exploration. Large HRNet/video fine-tuning may be impractical locally; choose later compute based on a measured pilot, without provisioning it now.

No performance, memory ceiling, or MPS training success is claimed before these checks run.

## 8. Reproducibility and evaluation record

Future runs use a unique run ID and a local ignored directory containing:

```text
config.resolved.yaml        # Complete arguments, not just overrides
provenance.json             # Git SHA/dirty diff hash, dataset release and splits
environment.txt            # Exact lock, Python/macOS/hardware/device/dtype
training.jsonl              # Losses, validation metrics, epochs, selected checkpoint
metrics.json                # Definitions, per-group metrics, intervals, coverage
predictions/                # Private IDs, timestamps, masks, errors; permission-aware
checkpoint/                 # Model, optimizer, scheduler, RNG and sampler state
```

Pin checkpoint hashes and upstream code commits. Persist annotation versions, deduplication decisions, exclusion counts, preprocessing/feature schemas, fitted normalization, selected stopping epoch, evaluation revision, and dataset license evidence. Save optimizer, scheduler, random-generator, and sampler states to support resume verification. A rerun must identify whether it is exact resume, fresh seeded training, or evaluation only.

Before substantive training, implement focused synthetic tests: group intersections including reuploads; augmentation coordinate inversion and left/right swaps; nearest-neighbor mask class preservation; 16-bit depth preservation and invalid pixels; masked losses and metric denominators; monotonic timestamps and frame-rate conversion; no future dependence in causal features/TCN; all-missing batches; cache invalidation after split/model changes; and checkpoint resume parity within documented device tolerance. Fixture data must be synthetic and local. Future CI runs these offline without weights or infant data.

For each experiment publish an aggregate table with protocol, modality, available/eligible groups, seed count, primary/secondary metrics, uncertainty, rejection rate, and limitations. Retain separate tables for official reproductions versus patient/group-clean protocols, manual ground truth versus pseudo-targets, and cross-domain evaluation. Failed runs and negative results remain in the experiment ledger.

## 9. Principal weaknesses and mitigations

| Risk | Consequence | Planned mitigation and remaining limit |
|---|---|---|
| No verified seizure events | No valid seizure-learning target or clinical evaluation | Keep present tasks nonclinical; a future study requires separately governed expert-adjudicated event annotations with appropriate clinical reference, potentially synchronized video-EEG |
| Sparse expression supervision | Landmark accuracy can be mistaken for expression understanding | Describe output as facial geometry; collect/validate a separate expression protocol before emotion claims |
| Small, selected, heterogeneous cohorts | Overfitting and uncertain population transfer | Grouped evaluation, simple baselines, uncertainty and per-group reporting; external validation remains necessary |
| Unknown identity / shared web sources | Patient leakage can survive video splitting | Conservative source groups and deduplication; explicitly limit claims when identities remain unknown |
| Depth versus RGB domain shift | A successful E3 does not supply a working RGB branch | Separate adapters/checkpoints and a gated RGB validation study |
| Predicted poses as targets | Temporal model can learn estimator artifacts | Label pseudo-target provenance, audit quality, use robustness analysis; independent manual video annotations are needed for anatomical validation |
| Camera motion, occlusion, caregiver intervention | Apparent infant motion may be spurious | Background-motion checks, validity masks, coverage and failure reports; exclusion does not prove robustness |
| Age, posture, skin appearance, clothing, setting | Shortcut learning or unequal performance | Record available metadata and stratify where sufficiently supported; do not infer sensitive attributes from appearance |
| Temporal downsampling and smoothing | Rapid motion may be aliased or removed | Preserve timestamps, declare bandwidth, compare preprocessing ablations; low-rate sources cannot recover missing dynamics |
| Separate task datasets | No complete paired multimodal gold standard | Evaluate branches independently and integration only on aligned observations; no claimed end-to-end clinical accuracy |
| Legacy code and weight terms | Porting errors or restricted artifacts | Minimal adapters, parity checks, pinned sources, per-artifact rights register |
| MPS variability and limited memory | Unsupported operations, inconsistent speed | CPU correctness reference, explicit device/fallback logs, small batches and cached features |

## 10. Implementation order and review gates

| Milestone | Deliverables after authorization | Completion evidence |
|---|---|---|
| M0 — current review | Documents, module/dataset directory scaffold, ignore rules | Links and scope consistency checked; **stop here** |
| M1 — environment + E0 | Minimal package, tested lock, synthetic fixtures, per-resource access/provenance audit | Offline synthetic checks and group/disjointness report; unresolved resources remain deferred |
| M2 — independent spatial experiments | E1/E2/E3 adapters and baseline runs, each separately | Validation-selected settings, appropriate held-out protocols, per-group errors and failures |
| M3 — temporal baseline | E4 feature schema, persistence/linear/TCN comparison | No future leakage, fixed pseudo-target definition, sealed-test reporting |
| M4 — integration | E5 aligned RGB branch/flow pipeline when gates pass | Coverage and synchronization evidence, upstream-training exclusion, common-subset ablations |
| M5 — transformer comparison | E6 frozen-video baseline and matched-input comparison | Equal evaluation groups, transparent compute and pretraining differences |
| Separate future clinical study | New data and annotation protocol, ethics/data governance, clinical collaborators | Clinically meaningful labels and independently designed validation; outside current project implementation |

No time-to-completion estimate is warranted before access and hardware pilots. The next user decision is review of the proposed protocols and priorities. **The present task ends with this scaffold and documentation; no implementation, installation, download, or training follows automatically.**
