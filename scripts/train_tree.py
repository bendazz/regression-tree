#!/usr/bin/env python3
"""
Train a DecisionTreeRegressor on the prepared California Housing CSV and export a compact JSON
for client-side visualization and prediction.

Outputs:
- model/tree.json : structure with nodes, thresholds, features, value, children
- model/meta.json : feature names, target, training stats, feature ranges/medians
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


def train_tree(
    df: pd.DataFrame,
    target_col: str,
    max_depth: int | None,
    min_samples_leaf: int,
    random_state: int,
) -> DecisionTreeRegressor:
    X = df.drop(columns=[target_col]).to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    tree.fit(X, y)
    return tree


def export_tree_json(tree: DecisionTreeRegressor, feature_names: List[str]) -> Dict[str, Any]:
    sk = tree.tree_
    nodes: List[Dict[str, Any]] = []

    def node_dict(i: int) -> Dict[str, Any]:
        is_leaf = sk.children_left[i] == sk.children_right[i]
        d: Dict[str, Any] = {
            "id": int(i),
            "isLeaf": bool(is_leaf),
            "nSamples": int(sk.n_node_samples[i]),
            "impurity": float(sk.impurity[i]),
            "value": float(sk.value[i][0][0]),
        }
        if not is_leaf:
            feat_idx = int(sk.feature[i])
            d.update(
                {
                    "feature": feature_names[feat_idx],
                    "featureIndex": feat_idx,
                    "threshold": float(sk.threshold[i]),
                    "left": int(sk.children_left[i]),
                    "right": int(sk.children_right[i]),
                }
            )
        return d

    for i in range(sk.node_count):
        nodes.append(node_dict(i))

    return {"root": 0, "nodes": nodes}


def main():
    parser = argparse.ArgumentParser(description="Train DecisionTreeRegressor and export JSON")
    parser.add_argument(
        "--train-csv",
        type=str,
        default="data/california_housing_train.csv",
        help="Path to training CSV",
    )
    parser.add_argument("--target", type=str, default="MedHouseVal", help="Target column")
    parser.add_argument("--max-depth", type=int, default=6, help="Max tree depth (None for unlimited)")
    parser.add_argument("--min-samples-leaf", type=int, default=20, help="Minimum samples per leaf")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="model")

    args = parser.parse_args()

    df = pd.read_csv(args.train_csv)
    feature_names = [c for c in df.columns if c != args.target]

    tree = train_tree(
        df=df,
        target_col=args.target,
        max_depth=None if args.max_depth <= 0 else args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )

    tree_json = export_tree_json(tree, feature_names)

    # Meta for client rendering/controls
    meta = {
        "featureNames": feature_names,
        "target": args.target,
        "nTrain": int(len(df)),
        "params": {
            "max_depth": None if args.max_depth <= 0 else args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "random_state": args.random_state,
        },
        "featureStats": {
            name: {
                "min": float(df[name].min()),
                "max": float(df[name].max()),
                "median": float(df[name].median()),
            }
            for name in feature_names
        },
        "targetStats": {
            "min": float(df[args.target].min()),
            "max": float(df[args.target].max()),
            "median": float(df[args.target].median()),
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tree.json").write_text(json.dumps(tree_json))
    (out_dir / "meta.json").write_text(json.dumps(meta))

    print(f"Exported model JSON to {out_dir}/tree.json and meta to {out_dir}/meta.json")


if __name__ == "__main__":
    main()
