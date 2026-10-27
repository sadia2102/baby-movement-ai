# InfAnFace image-audit environment

Verified on **27 October 2026**, CPython 3.13.3, macOS arm64, in the existing project `.venv`. The original [M1 environment record](m1-environment.md) and [M1 lock](macos-arm64-py313-dev.lock) are preserved.

Only **Pillow 13.0.0** was added. It is an optional `image-audit` extra, not a required core dependency. No NumPy, imagehash, OpenCV, model framework or network-tracking library is required for the implemented audit.

Selected wheel: `pillow-13.0.0-cp313-cp313-macosx_11_0_arm64.whl`.

SHA-256: `f11a3c1358bb3c68012a7b1f41b223c3db9f7bfde90b481df028cfe1c3363313`.

The wheel came from the [Pillow 13.0.0 PyPI release](https://pypi.org/project/pillow/13.0.0/), was checked against its published digest, and was installed from a local wheel directory using `--no-index --no-deps --require-hashes --only-binary=:all:`. A synthetic PNG encode/decode check and the complete synthetic image/provenance suite passed. The private dependency receipt records the exact wheel URL/hash. No source build was performed.

The [image-audit lock](macos-arm64-py313-image-audit.lock) combines the original nine tool pins with this one optional decoder. To reproduce on the same interpreter/platform, from the repository root:

```sh
.venv/bin/python -m pip --disable-pip-version-check --no-cache-dir install --only-binary=:all: --require-hashes -r environments/macos-arm64-py313-image-audit.lock
.venv/bin/python -m pip --disable-pip-version-check --no-cache-dir install --no-index --no-build-isolation --no-deps -e .
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m pip --disable-pip-version-check --no-cache-dir check
```

The first command explicitly downloads locked Python wheels if needed; it does not fetch research assets. For offline recreation, pre-obtain the exact wheels and use `--no-index --find-links /path/to/wheels` with the same hash lock. A different platform/interpreter needs its own verified wheel hashes.

**Results:** 150 tests passed with no skips, lint/format passed, dependency consistency passed. Without Pillow, audit-specific tests explicitly skip; a core-only M1 environment is not a full duplicate-audit verification. Audit utilities never install dependencies or download assets during imports/runs. No training, inference, MPS/CUDA compatibility or performance is claimed.

See the [dated duplicate/provenance report](../reports/infanface-duplicate-provenance-audit-2026-10-27.md) and README for methods, exact local commands and scientific limitations.
