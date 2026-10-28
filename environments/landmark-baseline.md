# Minimal CPU landmark baseline environment

Verified **28 October 2026** in separate `.venv-baseline`, created using the already available native **CPython 3.13.3**, macOS 27.0.1 arm64. System Python and the existing preprocessing `.venv` were not modified by an installer. Original environment records and historical locks remain unchanged. The project remains dependency-free at its core; the optional `baseline` extra declares the direct decoder/model dependencies.

The complete [baseline lock](macos-arm64-py313-baseline.lock) records **19 exact binary wheel versions and SHA-256 digests**. Every downloaded wheel was hashed and compared with its matching PyPI release JSON digest before installation. Installation then used only the local wheel directory, `--no-index --require-hashes --only-binary=:all:`. Editable project installation used `--no-index --no-build-isolation --no-deps`. The private `reports/private/baseline-dependencies-2026-10-28.json` records filenames, URLs, digests and wheel requirements.

| Purpose | Tested versions |
|---|---|
| Direct model/decoder | torch 2.14.1; Pillow 13.0.0 |
| Required Torch transitive dependencies | filelock 4.2.4; typing_extensions 4.16.0; sympy 1.14.0; networkx 3.7; Jinja2 3.1.6; fsspec 2026.9.0; mpmath 1.3.0; MarkupSafe 3.0.4 |
| Unchanged test/build pins, copied into separate environment | pip 25.0.1; setuptools 80.9.0; wheel 0.45.1; pytest 8.4.2; Ruff 0.14.2; iniconfig 2.3.0; packaging 25.0; pluggy 1.6.0; Pygments 2.19.2 |

The selected native model wheel is `torch-2.14.1-cp313-cp313-macosx_14_0_arm64.whl`, SHA-256 `dbe359d705f4d67236743794c296c6dee93a922fd8117eff8c3e880d7d0fb2b9`. Its approximately 127 MB is framework code, not pretrained weights. [Official PyPI release](https://pypi.org/project/torch/2.14.1/).

No torchvision, NumPy, pretrained checkpoints, additional datasets, CUDA-only libraries, notebook runtime or experiment tracker was installed. Torch emits a first-import warning about the unavailable optional NumPy bridge. No NumPy bridge is used; Pillow buffers, Torch tensors, optimizer/checkpoint operations and all synthetic tests work without it. This warning is recorded, not interpreted as a missing required dependency or silently suppressed. CPU only was used; no MPS/CUDA correctness or performance claim follows.

The existing environment is ready; run from the repository root:

```sh
.venv-baseline/bin/python -m pytest -q
.venv-baseline/bin/python -m ruff check .
.venv-baseline/bin/python -m ruff format --check .
.venv-baseline/bin/python -m pip --disable-pip-version-check --no-cache-dir check
.venv/bin/python -m pytest -q
.venv/bin/python -m pip --disable-pip-version-check --no-cache-dir check
```

Actual results: **248 passed, no skips** in the baseline environment; **224 passed, one optional Torch module skipped** in the original environment; lint and formatting (33 Python files) pass; both dependency checks pass. Installed distribution versions match their respective complete locks. The original 209 preprocessing tests also passed before changes began.

For a **fresh checkout with no `.venv-baseline`**, and Python 3.13.3 already available, reproduce explicitly:

```sh
python3.13 -m venv .venv-baseline
.venv-baseline/bin/python -m pip --disable-pip-version-check --no-cache-dir install --only-binary=:all: --require-hashes -r environments/macos-arm64-py313-baseline.lock
.venv-baseline/bin/python -m pip --disable-pip-version-check --no-cache-dir install --no-index --no-build-isolation --no-deps -e .
```

The first installation command explicitly downloads only locked Python wheels. For offline installation, obtain those exact wheels explicitly on an authorized connection, then add `--no-index --find-links /path/to/wheels`. No interpreter is downloaded; another interpreter/platform needs a separately resolved and verified lock. Do not install the baseline extra into the preserved preprocessing environment merely to run its original tools.

Synthetic CPU experiment, with a **new unused run name**:

```sh
.venv-baseline/bin/python scripts/verify_synthetic_baseline.py --config configs/experiments/synthetic_landmark_baseline.json --run-dir runs/synthetic-landmark-baseline-rerun
```

The fixed 32/16 manufactured-data task uses CPU float32, one thread, zero loader workers, deterministic algorithms and seed 42. It takes 480 optimizer steps, not an unbounded background job. Final selected validation pixel error is 0.22398287668670216, and all 16 reload predictions are identical. These are synthetic verification results only. Exact cross-platform/version reproducibility and exact interrupted-training resume remain unverified. See the [implementation report](../reports/landmark-baseline-synthetic-2026-10-28.md), [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.14/notes/randomness.html) and [checkpoint loader documentation](https://docs.pytorch.org/docs/2.14/generated/torch.load.html).
