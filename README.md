# regression-tree

Small utilities to prepare a regression dataset (California Housing) for an interactive decision tree visualization.

## Dataset prep

We use the California Housing dataset (sklearn) with a small test set for visualization.

Recommended sizes:
- Train: 3,000 rows
- Test: 300 rows

### Quick start

1) Install dependencies

```
pip install -r requirements.txt
```

2) Generate CSVs

```
python scripts/prepare_data.py --train-size 3000 --test-size 300
```

Outputs:
- `data/california_housing_train.csv`
- `data/california_housing_test.csv`

Flags:
- `--random-state` (default 42)
- `--bins` quantile bins for stratified split (default 10)
- `--out-dir` output directory (default `data`)

Columns include all features and the target `MedHouseVal`.