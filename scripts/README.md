# Project scripts

Run these commands from the repository root with `.venv\Scripts\python.exe`.

- `download_datasets.py`: download datasets with reachable public archives and
  validate all registered dataset loaders; reports manual setup for gated or
  otherwise non-automated sources.
- `cross_validate.py`, `few_shot_eval.py`: core evaluation entry points.
- `run_all_experiments.py` and `run_*_sweep.py` / `sweep_*.py`: experiment
  orchestration and parameter sweeps.
- `analysis/`: CSV merging/comparison and diagnostic helpers.

Scripts resolve the project root from their own location, so their internal
imports and subprocess calls do not depend on the shell's current directory.
Relative CLI paths are interpreted from the project root.
