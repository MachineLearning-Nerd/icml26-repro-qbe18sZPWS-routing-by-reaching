# Environment

- **Package manager:** `uv`.
- **Interpreter:** Python 3.10.18 (`.python-version` and `uv.lock`).
- **Recorded package set:** NumPy 1.26.4, Pytest 9.0.2, Torch 2.6.0, and
  TorchGFlowNet 2.4.1.
- **Compatibility overrides:** `torch-geometric==2.6.1` and Torch 2.6.0 are
  locked in `pyproject.toml` so the grid and molecule source trees share one
  environment. The deviation from the authors' README is documented in
  [`docs/SOURCE_AUDIT.md`](docs/SOURCE_AUDIT.md).
- **Compute:** the authoritative neural run used local Apple-silicon CPU; no
  GPU result is claimed. The lock records eight logical CPUs for the captured
  run.
- **Lock SHA-256:** `240aafac42d8dd4388c1f1e2239dbd5d530de02b3f0286232a5a06bc360c22e4`.

## Fixed command

```bash
uv sync --frozen
uv run python repro/src/run_campaign.py
```

The recorded cumulative run metadata is [`evidence/run_metadata.json`](evidence/run_metadata.json).
It reports the source run ID `fd92e2bc-ea94-4004-a820-62abf3e5e917`, exact,
neural, checker, molecule-prerequisite, direct-C6, and report-figure timings.
The raw full-grid and direct-C6 archives are verified by SHA-256 before use.
