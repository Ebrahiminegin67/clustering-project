
from __future__ import annotations
import logging
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

log = logging.getLogger(__name__)

def run_agglomerative(X: pd.DataFrame, n_clusters: int = 5, linkage: str = "single"):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X.values)
    return pd.Series(labels, index=X.index, name=f"agg_{linkage}"), model
