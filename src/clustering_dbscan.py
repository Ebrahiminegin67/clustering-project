
from __future__ import annotations
import logging
import pandas as pd
from sklearn.cluster import DBSCAN

log = logging.getLogger(__name__)

def run_dbscan(X: pd.DataFrame, eps: float = 0.7, min_samples: int = 5):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X.values)
    return pd.Series(labels, index=X.index, name="dbscan"), db
