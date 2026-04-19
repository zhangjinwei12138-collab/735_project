#!/usr/bin/env python3
"""
Run PCA and UMAP for single-cell expression data.

Expected input format:
1) counts CSV: first column is cell ID, remaining columns are gene counts.
2) metadata CSV: must contain cell ID column and cell-type column.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCA + UMAP for scRNA-seq counts matrix.")
    parser.add_argument("--counts", type=Path, required=True, help="Path to counts CSV.")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to metadata CSV.")
    parser.add_argument("--cell-col", type=str, default="cell", help="Cell ID column in metadata.")
    parser.add_argument(
        "--label-col",
        type=str,
        default="cell_ontology_class",
        help="Cell type/label column in metadata.",
    )
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--n-hvg", type=int, default=2000, help="Number of HVGs to keep.")
    parser.add_argument("--n-pcs", type=int, default=30, help="Number of PCs.")
    parser.add_argument("--target-sum", type=float, default=1e4, help="Library size target.")
    parser.add_argument(
        "--cluster-method",
        type=str,
        default="none",
        choices=["none", "gmm", "dp", "both"],
        help="Optional clustering method to run on embeddings.",
    )
    parser.add_argument(
        "--cluster-on",
        type=str,
        default="pca",
        choices=["pca", "umap"],
        help="Which embedding to cluster on.",
    )
    parser.add_argument("--n-clusters", type=int, default=7, help="Number of clusters for finite GMM.")
    parser.add_argument(
        "--dp-max-components",
        type=int,
        default=20,
        help="Max components for DP-like Bayesian GMM.",
    )
    parser.add_argument(
        "--dp-weight-concentration",
        type=float,
        default=1.0,
        help="Weight concentration prior for DP-like Bayesian GMM.",
    )
    parser.add_argument(
        "--covariance-type",
        type=str,
        default="full",
        choices=["full", "tied", "diag", "spherical"],
        help="Covariance type for mixture models.",
    )
    parser.add_argument("--gmm-max-iter", type=int, default=200, help="Max iterations for GMM models.")
    parser.add_argument("--gmm-n-init", type=int, default=5, help="Number of random initializations for GMM.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def select_hvg(log_expr: np.ndarray, gene_names: np.ndarray, n_hvg: int) -> tuple[np.ndarray, np.ndarray]:
    """Select top HVGs by dispersion (variance / mean) on log-normalized expression."""
    gene_means = log_expr.mean(axis=0)
    gene_vars = log_expr.var(axis=0)
    dispersion = gene_vars / (gene_means + 1e-8)
    valid = np.isfinite(dispersion) & (gene_means > 0)

    if valid.sum() == 0:
        raise ValueError("No valid genes found for HVG selection.")

    valid_idx = np.where(valid)[0]
    valid_disp = dispersion[valid]
    keep = min(n_hvg, valid_idx.size)
    top_local = np.argpartition(valid_disp, -keep)[-keep:]
    top_idx = valid_idx[top_local]
    top_idx = top_idx[np.argsort(dispersion[top_idx])[::-1]]

    return log_expr[:, top_idx], gene_names[top_idx]


def plot_embedding(
    emb: np.ndarray,
    labels: pd.Series,
    x_name: str,
    y_name: str,
    title: str,
    out_png: Path,
) -> None:
    plt.figure(figsize=(8, 6))
    label_values = labels.astype(str).values
    unique_labels = pd.unique(label_values)

    for lab in unique_labels:
        idx = label_values == lab
        plt.scatter(emb[idx, 0], emb[idx, 1], s=10, alpha=0.8, label=lab)

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    plt.legend(loc="best", markerscale=1.5, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def safe_metrics(true_labels: np.ndarray, pred_labels: np.ndarray, emb: np.ndarray) -> tuple[float, float]:
    ari = float(adjusted_rand_score(true_labels, pred_labels))
    n_clusters = np.unique(pred_labels).size
    if 1 < n_clusters < emb.shape[0]:
        try:
            sil = float(silhouette_score(emb, pred_labels))
        except ValueError:
            sil = float("nan")
    else:
        sil = float("nan")
    return ari, sil


def run_clustering(
    emb: np.ndarray,
    true_labels: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict[str, float | int | str]], dict[str, np.ndarray]]:
    metrics_rows: list[dict[str, float | int | str]] = []
    assignments: dict[str, np.ndarray] = {}

    methods: list[str]
    if args.cluster_method == "both":
        methods = ["gmm", "dp"]
    elif args.cluster_method == "none":
        methods = []
    else:
        methods = [args.cluster_method]

    for method in methods:
        if method == "gmm":
            model = GaussianMixture(
                n_components=args.n_clusters,
                covariance_type=args.covariance_type,
                max_iter=args.gmm_max_iter,
                n_init=args.gmm_n_init,
                random_state=args.seed,
            )
            pred = model.fit_predict(emb)
            ari, sil = safe_metrics(true_labels, pred, emb)
            metrics_rows.append(
                {
                    "method": "gmm",
                    "space": args.cluster_on,
                    "n_clusters_param": args.n_clusters,
                    "n_clusters_found": int(np.unique(pred).size),
                    "ari": ari,
                    "silhouette": sil,
                    "bic": float(model.bic(emb)),
                }
            )
            assignments["gmm"] = pred
        elif method == "dp":
            model = BayesianGaussianMixture(
                n_components=args.dp_max_components,
                covariance_type=args.covariance_type,
                weight_concentration_prior_type="dirichlet_process",
                weight_concentration_prior=args.dp_weight_concentration,
                max_iter=args.gmm_max_iter,
                n_init=max(1, args.gmm_n_init),
                random_state=args.seed,
            )
            pred = model.fit_predict(emb)
            ari, sil = safe_metrics(true_labels, pred, emb)
            metrics_rows.append(
                {
                    "method": "dp",
                    "space": args.cluster_on,
                    "n_clusters_param": args.dp_max_components,
                    "n_clusters_found": int(np.unique(pred).size),
                    "ari": ari,
                    "silhouette": sil,
                    "bic": float("nan"),
                }
            )
            assignments["dp"] = pred

    return metrics_rows, assignments


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    counts_df = pd.read_csv(args.counts)
    if counts_df.shape[1] < 2:
        raise ValueError("Counts CSV must have at least one cell ID column and one gene column.")

    cell_ids = counts_df.iloc[:, 0].astype(str)
    gene_names = counts_df.columns[1:].to_numpy()
    expr = counts_df.iloc[:, 1:].to_numpy(dtype=np.float32)

    metadata_df = pd.read_csv(args.metadata)
    for col in [args.cell_col, args.label_col]:
        if col not in metadata_df.columns:
            raise ValueError(f"Column '{col}' not found in metadata.")

    metadata_df[args.cell_col] = metadata_df[args.cell_col].astype(str)
    metadata_df = metadata_df.set_index(args.cell_col)

    if not pd.Index(cell_ids).isin(metadata_df.index).all():
        missing = (~pd.Index(cell_ids).isin(metadata_df.index)).sum()
        raise ValueError(f"{missing} cells in counts are missing in metadata.")

    labels = metadata_df.loc[cell_ids, args.label_col]

    # Library-size normalization + log1p transform.
    libsize = expr.sum(axis=1, keepdims=True)
    libsize[libsize == 0] = 1.0
    expr_norm = (expr / libsize) * args.target_sum
    expr_log = np.log1p(expr_norm)

    if args.n_hvg > 0:
        expr_use, hvg_names = select_hvg(expr_log, gene_names, args.n_hvg)
    else:
        expr_use, hvg_names = expr_log, gene_names

    n_pcs = min(args.n_pcs, expr_use.shape[0] - 1, expr_use.shape[1])
    if n_pcs < 2:
        raise ValueError("Not enough data to compute PCA with at least 2 components.")

    pca_model = PCA(n_components=n_pcs, random_state=args.seed)
    pca_emb = pca_model.fit_transform(expr_use)

    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.3,
        metric="euclidean",
        random_state=args.seed,
    )
    umap_emb = umap_model.fit_transform(pca_emb)

    pca_cols = [f"PC{i+1}" for i in range(pca_emb.shape[1])]
    pca_out = pd.DataFrame(pca_emb, columns=pca_cols)
    pca_out.insert(0, "cell", cell_ids.values)
    pca_out["label"] = labels.values
    pca_out.to_csv(args.outdir / "pca_embedding.csv", index=False)

    umap_out = pd.DataFrame(
        {
            "cell": cell_ids.values,
            "label": labels.values,
            "UMAP1": umap_emb[:, 0],
            "UMAP2": umap_emb[:, 1],
        }
    )
    umap_out.to_csv(args.outdir / "umap_embedding.csv", index=False)

    pd.Series(hvg_names, name="gene").to_csv(args.outdir / "hvg_genes.csv", index=False)

    evr = pca_model.explained_variance_ratio_
    evr_df = pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(evr))], "explained_variance_ratio": evr})
    evr_df.to_csv(args.outdir / "pca_explained_variance_ratio.csv", index=False)

    plot_embedding(
        pca_emb[:, :2],
        labels,
        "PC1",
        "PC2",
        "PCA (first 2 components)",
        args.outdir / "pca_plot.png",
    )
    plot_embedding(
        umap_emb,
        labels,
        "UMAP1",
        "UMAP2",
        "UMAP (from PCA space)",
        args.outdir / "umap_plot.png",
    )

    if args.cluster_on == "pca":
        cluster_emb = pca_emb
    else:
        cluster_emb = umap_emb

    metrics_rows, assignments = run_clustering(cluster_emb, labels.astype(str).values, args)
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(args.outdir / f"clustering_metrics_{args.cluster_on}.csv", index=False)
        for method, pred in assignments.items():
            out_df = pd.DataFrame(
                {
                    "cell": cell_ids.values,
                    "true_label": labels.values,
                    "cluster": pred,
                }
            )
            out_df.to_csv(args.outdir / f"cluster_assignments_{method}_{args.cluster_on}.csv", index=False)
            plot_embedding(
                cluster_emb[:, :2],
                pd.Series(pred.astype(str)),
                f"{method.upper()} clusters on {args.cluster_on.upper()}",
                f"{args.cluster_on.upper()}1",
                f"{args.cluster_on.upper()}2",
                args.outdir / f"cluster_plot_{method}_{args.cluster_on}.png",
            )

    print(f"Done. Output directory: {args.outdir}")
    print(f"Cells: {expr.shape[0]}, genes: {expr.shape[1]}, HVGs used: {expr_use.shape[1]}")
    print(f"PCA components: {pca_emb.shape[1]}")
    print("Saved files: pca_embedding.csv, umap_embedding.csv, hvg_genes.csv,")
    print("             pca_explained_variance_ratio.csv, pca_plot.png, umap_plot.png")
    if metrics_rows:
        print("             clustering_metrics_<space>.csv, cluster_assignments_*.csv, cluster_plot_*.png")


if __name__ == "__main__":
    main()
