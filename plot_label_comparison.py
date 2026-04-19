#!/usr/bin/env python3
"""
Plot true labels and six clustered labels side-by-side:
- PCA + GMM
- PCA + DP
- UMAP + GMM
- UMAP + DP
- AE(latent) + GMM
- AE(latent) + DP
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare true labels with 6 clustering outputs.")
    parser.add_argument("--pca-dir", type=Path, required=True, help="Directory from run_pca_umap.py output.")
    parser.add_argument("--ae-dir", type=Path, required=True, help="Directory from run_autoencoder.py output.")
    parser.add_argument("--outdir", type=Path, default=None, help="Output directory. Default: pca-dir.")
    parser.add_argument(
        "--viz-space",
        type=str,
        default="umap",
        choices=["umap", "pca", "ae"],
        help="Shared 2D coordinates for all panels.",
    )
    parser.add_argument("--point-size", type=float, default=7.0, help="Scatter point size.")
    return parser.parse_args()


def load_viz_coords(args: argparse.Namespace) -> pd.DataFrame:
    if args.viz_space == "umap":
        path = args.pca_dir / "umap_embedding.csv"
        df = pd.read_csv(path)
        return df[["cell", "label", "UMAP1", "UMAP2"]].rename(columns={"UMAP1": "x", "UMAP2": "y"})
    if args.viz_space == "pca":
        path = args.pca_dir / "pca_embedding.csv"
        df = pd.read_csv(path)
        return df[["cell", "label", "PC1", "PC2"]].rename(columns={"PC1": "x", "PC2": "y"})
    path = args.ae_dir / "autoencoder_latent.csv"
    df = pd.read_csv(path)
    z_cols = [c for c in df.columns if c.startswith("z")]
    if len(z_cols) < 2:
        raise ValueError(f"Need at least 2 latent dimensions in {path}.")
    return df[["cell", "label", z_cols[0], z_cols[1]]].rename(columns={z_cols[0]: "x", z_cols[1]: "y"})


def read_assignment(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing assignment file for {name}: {path}")
    df = pd.read_csv(path)
    if not {"cell", "cluster"}.issubset(set(df.columns)):
        raise ValueError(f"Assignment file has wrong schema: {path}")
    return df[["cell", "cluster"]].rename(columns={"cluster": name})


def scatter_categorical(ax: plt.Axes, x: np.ndarray, y: np.ndarray, labels: np.ndarray, s: float) -> int:
    codes = pd.Categorical(labels.astype(str))
    n_classes = len(codes.categories)
    cmap = "tab20" if n_classes <= 20 else "tab20b"
    ax.scatter(x, y, c=codes.codes, cmap=cmap, s=s, alpha=0.85, linewidths=0)
    return n_classes


def main() -> None:
    args = parse_args()
    outdir = args.outdir if args.outdir is not None else args.pca_dir
    outdir.mkdir(parents=True, exist_ok=True)

    base = load_viz_coords(args)
    base["cell"] = base["cell"].astype(str)
    base["label"] = base["label"].astype(str)

    method_files = {
        "pca_gmm": args.pca_dir / "cluster_assignments_gmm_pca.csv",
        "pca_dp": args.pca_dir / "cluster_assignments_dp_pca.csv",
        "umap_gmm": args.pca_dir / "cluster_assignments_gmm_umap.csv",
        "umap_dp": args.pca_dir / "cluster_assignments_dp_umap.csv",
        "ae_gmm": args.ae_dir / "cluster_assignments_gmm_latent.csv",
        "ae_dp": args.ae_dir / "cluster_assignments_dp_latent.csv",
    }

    merged = base.copy()
    for name, path in method_files.items():
        assign_df = read_assignment(path, name)
        assign_df["cell"] = assign_df["cell"].astype(str)
        merged = merged.merge(assign_df, on="cell", how="left")

    panel_order = [
        ("true_label", "label"),
        ("PCA + GMM", "pca_gmm"),
        ("PCA + DP", "pca_dp"),
        ("UMAP + GMM", "umap_gmm"),
        ("UMAP + DP", "umap_dp"),
        ("AE + GMM", "ae_gmm"),
        ("AE + DP", "ae_dp"),
    ]

    metric_rows: list[dict[str, float | int | str]] = []
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), dpi=140)
    flat_axes = axes.ravel()

    for i, (title, col) in enumerate(panel_order):
        ax = flat_axes[i]
        y = merged[col].astype(str).to_numpy()
        n_classes = scatter_categorical(ax, merged["x"].to_numpy(), merged["y"].to_numpy(), y, s=args.point_size)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"{args.viz_space.upper()}1")
        ax.set_ylabel(f"{args.viz_space.upper()}2")
        ax.set_xticks([])
        ax.set_yticks([])

        if col == "label":
            ari = float("nan")
        else:
            ari = float(adjusted_rand_score(merged["label"].astype(str).to_numpy(), y))
            ax.set_title(f"{title}\nARI={ari:.3f}, clusters={n_classes}", fontsize=11)

        metric_rows.append({"panel": title, "column": col, "ari_vs_true": ari, "n_clusters_displayed": n_classes})

    for j in range(len(panel_order), len(flat_axes)):
        flat_axes[j].axis("off")

    fig.suptitle(f"True Label vs 6 Clustered Labels (viz space: {args.viz_space})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = outdir / f"label_comparison_{args.viz_space}.png"
    fig.savefig(fig_path)
    plt.close(fig)

    metric_df = pd.DataFrame(metric_rows)
    metric_path = outdir / f"label_comparison_metrics_{args.viz_space}.csv"
    metric_df.to_csv(metric_path, index=False)

    print(f"Saved figure: {fig_path}")
    print(f"Saved metrics: {metric_path}")


if __name__ == "__main__":
    main()
