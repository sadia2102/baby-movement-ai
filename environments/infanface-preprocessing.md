# InfAnFace preprocessing environment

Verified **28 October 2026**, using the existing project `.venv`: CPython **3.13.3**, native macOS arm64, Pillow **13.0.0**. No package, interpreter, extra, system setting or lock entry was installed or changed for this milestone. The package still has zero required runtime dependencies; image loading/preprocessing uses the existing optional `image-audit` extra. Pure affine geometry has no decoder dependency.

All installed distribution versions named by the existing [development lock](macos-arm64-py313-dev.lock) and [image-audit lock](macos-arm64-py313-image-audit.lock) matched their pins. `pip check` reported no broken requirements. The previous [decoder installation record](infanface-image-audit.md), selected wheel hash and [M1 record](m1-environment.md) remain unchanged; there was no new wheel acquisition to verify. The archived Pillow wheel SHA-256 remains `f11a3c1358bb3c68012a7b1f41b223c3db9f7bfde90b481df028cfe1c3363313`.

Run from the repository root with the existing environment, without installation or network access:

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m pip --disable-pip-version-check --no-cache-dir check
.venv/bin/infant-movement audit tests/fixtures/synthetic_metadata.json
```

Results: **209 tests passed with no skips**, including 59 new preprocessing tests. Lint passed, all 23 Python files were formatted, dependency consistency passed, and the synthetic metadata example returned 0 with its expected unknown-subject limitation. The optional image tests skip if Pillow is absent; a core-only environment does not verify this milestone completely.

The separate local-data smoke checks used fixed 256×256 full-raster inputs under explicit unverified base-0/stored-annotation assumptions. Stored policy returned 0 with 410 processed; EXIF normalization returned 1 with 409 processed and one invalid orientation tag rejected. Neither operation fit preprocessing or ran a model. See [the dated report](../reports/infanface-preprocessing-2026-10-28.md) and README for exact commands, coordinate conventions and limitations.

No PyTorch, torchvision, NumPy, model weights, MPS/CUDA checks or training environment were introduced. Other interpreters/platforms and compatibility with upstream model preprocessing remain UNVERIFIED.
