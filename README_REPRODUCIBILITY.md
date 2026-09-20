# ECG arrhythmia — reproducibility package

Generated: 2026-09-19T23:07:34

## Contents

- `source/`: source code and configuration used by the experiments.
- `results/tables/`: CSV and LaTeX result tables.
- `results/figures/`: figures, including the paper figures and interpretability figure when available.
- `results/models/`: trained Keras checkpoints (included: True).
- `notebooks/`: notebooks found in the Colab `/content` directory.
- `environment.txt`: main package versions.
- `DATASET_NOTICE.txt`: dataset licensing and download notice.

## Important

The MIT-BIH database is deliberately not redistributed here. Obtain it from PhysioNet and provide its local path to the scripts.
The result tables and figures are copied from the completed experiment; this package does not include the raw database.

## Reproduction order

1. Install the packages listed in `environment.txt`.
2. Download the MIT-BIH Arrhythmia Database from PhysioNet.
3. Open the notebook in `notebooks/` or inspect the scripts in `source/`.
4. Reproduce the train/test split before running any model.
5. Run the evaluation and figure-generation scripts.

## Final report assets

The principal paper figures are under `results/figures/paper/`.
The principal result tables are under `results/tables/`.
