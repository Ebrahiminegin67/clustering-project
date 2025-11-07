
"""Data loading utilities for Mall Customers or any CSV URL/local path."""
from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd

log = logging.getLogger(__name__)

IBM_URL = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/Cust_Segmentation.csv"

def load_csv(path_or_url: str) -> pd.DataFrame:
    src = str(path_or_url)
    if src.startswith("http://") or src.startswith("https://"):
        df = pd.read_csv(src)
        log.info("Loaded remote CSV: %s shape=%s", src, df.shape)
        return df
    path = Path(src)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    log.info("Loaded local CSV: %s shape=%s", path.name, df.shape)
    return df

def basic_info(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "na_counts": df.isna().sum().to_dict(),
        "describe_numeric": df.describe(include="number").to_dict(),
    }
