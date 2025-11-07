
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def scatter_2d(Z: pd.DataFrame, labels: pd.Series, title: str, outdir: Path, fname: str) -> Path:
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / fname
    plt.figure(figsize=(6,5))
    plt.scatter(Z.iloc[:,0], Z.iloc[:,1], c=labels, s=25)
    plt.title(title); plt.xlabel(Z.columns[0]); plt.ylabel(Z.columns[1])
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path
