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

## Train model and run locally

```
python scripts/train_tree.py --out-dir model --max-depth 6 --min-samples-leaf 20
python -m http.server 8000
```

Open http://localhost:8000/ (root index) or http://localhost:8000/web/ (dev index).

## Deploy to GitHub Pages

- Commit the generated `model/` and `data/` directories along with `index.html` at the repo root.
- Enable GitHub Pages (Settings → Pages) with source “Deploy from a branch”, branch `main`, folder `/root`.
- Your site will load assets via relative paths that work on Pages.