
from __future__ import annotations
import logging
import pandas as pd
from sklearn.cluster import KMeans

log = logging.getLogger(__name__)

def run_kmeans(X: pd.DataFrame, n_clusters: int = 5, random_state: int = 42):
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(X.values)
    return pd.Series(labels, index=X.index, name="kmeans"), km
