
from __future__ import annotations
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score

def silhouette(X: pd.DataFrame, labels: pd.Series) -> float:
    if len(set(labels)) < 2 or set(labels) == {-1}:
        return float("nan")
    return float(silhouette_score(X.values, labels.values))

def rand_index(true_labels: pd.Series, pred_labels: pd.Series) -> float:
    return float(adjusted_rand_score(true_labels.values, pred_labels.values))

def purity(true_labels: pd.Series, pred_labels: pd.Series) -> float:
    cm = pd.crosstab(true_labels, pred_labels)
    return float(cm.max(axis=0).sum() / cm.values.sum())

def evaluate_all(X: pd.DataFrame, labels_dict: dict[str, pd.Series], true_labels: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    for name, lab in labels_dict.items():
        row = {"model": name, "silhouette": silhouette(X, lab)}
        if true_labels is not None:
            row["adj_rand"] = rand_index(true_labels, lab)
            row["purity"] = purity(true_labels, lab)
        rows.append(row)
    return pd.DataFrame(rows)
