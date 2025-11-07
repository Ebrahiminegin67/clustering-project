
from __future__ import annotations
import argparse, logging
from pathlib import Path
import pandas as pd
from .data_loader import load_csv, IBM_URL, basic_info
from .preprocessing import prepare_features, impute_numeric, scale_features, pca_project
from .clustering_kmeans import run_kmeans
from .clustering_agglomerative import run_agglomerative
from .clustering_dbscan import run_dbscan
from .evaluation import evaluate_all
from .plotting import scatter_2d

def setup_logging(level: str = "INFO"):
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

def parse_args():
    p = argparse.ArgumentParser(description="Clustering lab pipeline")
    p.add_argument("--data", type=str, default="data/Mall_Customers.csv", help="CSV path or URL")
    p.add_argument("--download-ibm", action="store_true", help="Use IBM public URL for Mall Customers")
    p.add_argument("--k", type=int, default=5, help="Number of clusters for KMeans & Agglomerative")
    p.add_argument("--linkages", type=str, nargs="*", default=["single","complete"], help="Agglomerative linkages")
    p.add_argument("--eps", type=float, default=0.7, help="DBSCAN eps")
    p.add_argument("--min_samples", type=int, default=5, help="DBSCAN min_samples")
    p.add_argument("--out", type=str, default="outputs", help="Output directory")
    p.add_argument("--log", type=str, default="INFO", help="Logging level")
    return p.parse_args()

def main():
    args = parse_args()
    setup_logging(args.log)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    src = IBM_URL if args.download_ibm else args.data
    df = load_csv(src)
    X = prepare_features(df)
    X = impute_numeric(X, strategy="median")
    Xs, _ = scale_features(X)
    Z, _ = pca_project(Xs, n_components=2)

    labels_dict = {}

    # KMeans
    km_labels, _ = run_kmeans(Xs, n_clusters=args.k)
    labels_dict[km_labels.name] = km_labels
    scatter_2d(Z, km_labels, f"KMeans (k={args.k})", outdir, "kmeans_clusters.png")

    # Agglomerative
    for link in args.linkages:
        agg_labels, _ = run_agglomerative(Xs, n_clusters=args.k, linkage=link)
        labels_dict[agg_labels.name] = agg_labels
        scatter_2d(Z, agg_labels, f"Agglomerative ({link})", outdir, f"agg_{link}_clusters.png")

    # DBSCAN
    db_labels, _ = run_dbscan(Xs, eps=args.eps, min_samples=args.min_samples)
    labels_dict[db_labels.name] = db_labels
    scatter_2d(Z, db_labels, f"DBSCAN (eps={args.eps}, min_samples={args.min_samples})", outdir, "dbscan_clusters.png")

    eval_df = evaluate_all(Xs, labels_dict, true_labels=None)
    eval_df.to_csv(outdir / "silhouette_scores.csv", index=False)

    print(f"Done. Outputs saved to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
