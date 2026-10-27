# M1 environment and verification record

Verified **27 October 2026** for metadata validation and grouped split audits only. The original research plan remains unchanged. These checks use synthetic fixtures; no real infant data, weights, training, or inference is involved.

## Environment

| Item | Observed value |
|---|---|
| Python | CPython 3.13.3, native arm64 |
| Python compiler | Clang 16.0.0 (clang-1600.0.26.6) |
| OS | macOS 27.0.1 |
| Isolation | Project-local `.venv`; system site packages disabled |
| Resolver/installer | pip 25.0.1 |
| Build | setuptools 80.9.0, wheel 0.45.1 |
| Development | pytest 8.4.2, Ruff 0.14.2 |
| Transitive tools | iniconfig 2.3.0, packaging 25.0, pluggy 1.6.0, Pygments 2.19.2 |
| Local project | infant-movement 0.1.0, editable install |
| Git | 2.54.0 (Apple Git-157); local `main`, no commits or remotes created |
| Runtime dependencies | None; Python standard library only |

[macos-arm64-py313-dev.lock](macos-arm64-py313-dev.lock) records all nine tool distributions with SHA-256 hashes of the exact wheels obtained from PyPI. Ruff's wheel is macOS arm64; the other locked wheels are platform-independent. Dependencies were downloaded explicitly, then installed from the local wheel directory with `--no-index --require-hashes --only-binary=:all:`. The existing pip 25.0.1 satisfied its pin; a clean-environment recreation also installs/checks the locked tools. No Python interpreter was downloaded or system installation changed. The editable project was installed offline with `--no-build-isolation --no-deps` after installing the build tools.

The plan's proposed Python 3.11 training environment was not provisioned. No PyTorch/MPS/CUDA compatibility or performance is claimed. Python >=3.11 in package metadata states an intended minimum; only 3.13.3 on this platform was exercised. Regenerate and verify a separate lock for another platform or interpreter.

## Checks performed

| Check | Result |
|---|---|
| `python -m pytest -q` | 82 passed; synthetic data only |
| `python -m ruff check .` | Passed |
| `python -m ruff format --check .` | Passed |
| `python -m pip --disable-pip-version-check --no-cache-dir check` | No broken requirements |
| Editable package installation | Passed; importable from outside the repository |
| `infant-movement audit tests/fixtures/synthetic_metadata.json` | Exit 0, 3 records, one per partition; one unknown-subject warning |
| CLI negative paths | Synthetic leakage exit 1; invalid/missing input exit 2; inputs unchanged |
| Wheel build and separate installation | Passed in a fresh temporary venv; full synthetic suite and installed command verified outside checkout |
| Git ignore boundaries | Environment, data assets, weights, caches, private manifests/reports, and build output excluded; original directory markers retained |
| Original plan integrity | SHA-256 `ba5424d9e1ba1f41c6edf4c9a759ec5cd29cc01ea74c474153d277e8114532ad` unchanged |

The wheel verification uses a separate temporary environment with the same hash-locked tools, builds without dependency/network resolution, installs the built wheel without runtime dependencies, and runs the same suite against that installation. No source-path override is used by pytest. The CLI is checked from outside the checkout. The maintained development environment remains `.venv`.

## Limits and next boundary

There are no experimental run records, scientific results, real-data split reports, or clinical claims. Registry evidence, patient independence, unknown cross-dataset overlap, data/media rights, annotation validity, and upstream-training membership remain **UNVERIFIED**. A passing audit establishes consistency of supplied keys and assignments only. Metadata timestamps are checked for numeric validity, not sequence monotonicity or alignment to real recordings. Coordinate, depth, label-mask, model, causal-window, and metric tests are deferred until their implementations are authorized.

M1 ends here. Review the metadata contract and resolve one approved resource's access/provenance requirements before any acquisition or adapter implementation. No subsequent milestone is authorized by this record.
