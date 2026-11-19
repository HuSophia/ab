# MR-PCSB: five-independent-training-split replication

This package reruns the paper's three **analytic OU bridge** variants on five fresh APEBench Kolmogorov simulation splits. Only the train/test generation seeds differ from the original protocol. The original train/test pair is excluded, no hyperparameters are selected on these splits, and no neural model is trained.

## Run the official experiment

Use an environment with Python 3.12, package-install access, and APEBench's CPU dependencies:

```bash
python -m pip install -r requirements-apebench.txt
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python mrpcsb_replication.py --profile study --out new-official-results --discard-inputs
```

The output directory must not exist. The five pairs are fixed in `mrpcsb_replication.py`: train seeds 51001–51005, test seeds 91001–91005. The driver starts a fresh official generator for each split, then computes summaries in the same process. The endpoint shuffle, training-only reference calibration, low-mode coordinates, four-way partitions, Solver settings, and diagnostics are unchanged. A solver or data check failure stops the run and preserves completed split output; do not silently substitute another seed.

Per split: 64 training trajectories used only as independently shuffled endpoint populations; 160 test trajectories used with their matching records, grouped into five disjoint batches of 32. Group metrics are first averaged within each split, then split scores are paired by train/test draw across methods. Therefore the uncertainty unit is one independent training/test experiment, not a hold-out group or a trajectory. The 95% paired t intervals are descriptive, unadjusted, and based on five independent splits.

- `report.md`: aggregate results and interpretation boundaries.
- `objectives_by_split.csv`: fine objective gaps, proportion recovered, primal–dual numerical brackets, constraints, and construction times.
- `scores_by_split.csv`: per-method metrics on each split.
- `paired_differences_by_split.csv`: per-split paired effects.
- `paired_summary.csv`: five-split means, intervals, and effect-direction counts.
- `manifest.json`: exact dependency versions, Python/platform, source hashes, split seeds, and input hashes.
- `split_*/`: generation and numerical metadata. With `--discard-inputs`, each coordinate archive is deleted after scoring.

The optional GitHub Actions template is customized for the `HuSophia/ab` project branch. It assumes the package is in `mrpcsb_study/replication` and publishes only the aggregate Markdown and CSV files, with no uploaded checkpoints or trajectory archives. It is not needed for local or Colab runs.

## Test without APEBench

The five-split synthetic OU fixture checks I/O, metrics, rounding, and paired aggregation under the same shapes. It is **not** a fluid experiment and must not be cited as empirical evidence. In an environment with NumPy and SciPy:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1   python mrpcsb_replication.py --profile fixture --out new-fixture-run --discard-inputs
OPENBLAS_NUM_THREADS=1 python block_relaxation_primal_qa.py
```
