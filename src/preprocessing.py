
"""Preprocessing: numeric selection, scaling, optional PCA."""

from __future__ import annotations
import logging
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

log = logging.getLogger(__name__)

def prepare_features(df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    if feature_cols is None:
        feature_cols = df.select_dtypes(include="number").columns.tolist()
    X = df[feature_cols].copy()
    log.info("Selected %d numeric features.", len(feature_cols))
    return X

def impute_numeric(X: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Impute NaNs in numeric features (default: median)."""
    imp = SimpleImputer(strategy=strategy)
    Xi = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)
    n_before = int(X.isna().sum().sum())
    n_after = int(Xi.isna().sum().sum())
    log.info("Imputed numeric NaNs: %d -> %d (strategy=%s)", n_before, n_after, strategy)
    return Xi

def scale_features(X: pd.DataFrame):
    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    log.info("Standardized features.")
    return Xs, scaler

def pca_project(X: pd.DataFrame, n_components: int = 2):
    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(X.values)
    cols = [f"PC{i}" for i in range(1, n_components+1)]
    return pd.DataFrame(Z, columns=cols, index=X.index), pca
