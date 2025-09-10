#!/usr/bin/env python3
"""
Prepare California Housing regression dataset for interactive visualization.

Features:
- Fetches dataset via sklearn (as pandas DataFrame)
- Stratified shuffle split using binned target to preserve distribution
- Allows fixed train/test sizes (small test for viz), reproducible with seed
- Writes CSVs to data/ directory

Usage (defaults: train=3000, test=300):
  python scripts/prepare_data.py
  python scripts/prepare_data.py --train-size 5000 --test-size 500 --random-state 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import StratifiedShuffleSplit


def load_california_housing() -> pd.DataFrame:
    data = fetch_california_housing(as_frame=True)
    df: pd.DataFrame = data.frame.copy()
    # Keep column names as-is; target is 'MedHouseVal'
    return df


def make_stratify_bins(y: pd.Series, n_bins: int = 10) -> np.ndarray:
    """Bin continuous target into quantiles for stratification.

    Handles duplicate edges by dropping and using resulting number of bins.
    Returns integer bin codes suitable for stratify parameter.
    """
    # Use qcut to create (approx) equal-frequency bins
    binned = pd.qcut(y, q=n_bins, duplicates="drop")
    return binned.astype("category").cat.codes.to_numpy()


def stratified_fixed_sizes(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: int,
    test_size: int,
    random_state: int = 42,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return train/test DataFrames with fixed row counts using stratified shuffle.

    Unused rows are dropped (we only keep requested sizes for snappy viz).
    """
    n_total = len(X)
    assert train_size > 0 and test_size > 0, "train_size and test_size must be > 0"
    assert train_size + test_size <= n_total, (
        f"Requested sizes exceed dataset: {train_size + test_size} > {n_total}"
    )

    stratify_bins = make_stratify_bins(y, n_bins=n_bins)

    train_prop = train_size / n_total
    test_prop = test_size / n_total

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train_prop,
        test_size=test_prop,
        random_state=random_state,
    )

    # Use first split
    indices = np.arange(n_total)
    train_idx, test_idx = next(splitter.split(indices.reshape(-1, 1), stratify_bins))

    X_y = X.copy()
    X_y[y.name] = y

    train_df = X_y.iloc[train_idx].reset_index(drop=True)
    test_df = X_y.iloc[test_idx].reset_index(drop=True)

    # As a safety check, ensure sizes match exactly (within 1 due to rounding) and, if needed, trim.
    if len(train_df) > train_size:
        train_df = train_df.sample(n=train_size, random_state=random_state).reset_index(drop=True)
    if len(test_df) > test_size:
        test_df = test_df.sample(n=test_size, random_state=random_state + 1).reset_index(drop=True)

    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Prepare California Housing regression dataset")
    parser.add_argument("--train-size", type=int, default=3000, help="Number of training rows")
    parser.add_argument("--test-size", type=int, default=300, help="Number of test rows")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--bins", type=int, default=10, help="Number of quantile bins for stratification")
    parser.add_argument(
        "--out-dir", type=str, default="data", help="Output directory for CSV files"
    )
    args = parser.parse_args()

    df = load_california_housing()
    feature_cols = [c for c in df.columns if c != "MedHouseVal"]
    target = df["MedHouseVal"]
    features = df[feature_cols]

    if args.train_size + args.test_size > len(df):
        raise SystemExit(
            f"Requested train+test = {args.train_size + args.test_size} exceeds dataset size {len(df)}"
        )

    train_df, test_df = stratified_fixed_sizes(
        X=features,
        y=target,
        train_size=args.train_size,
        test_size=args.test_size,
        random_state=args.random_state,
        n_bins=args.bins,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "california_housing_train.csv"
    test_path = out_dir / "california_housing_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(
        f"Wrote {len(train_df)} rows to {train_path} and {len(test_df)} rows to {test_path}."
    )


if __name__ == "__main__":
    main()
