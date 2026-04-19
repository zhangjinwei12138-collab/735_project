#!/usr/bin/env python3
"""
Train an autoencoder on single-cell counts and export latent embeddings.

Input format:
1) counts CSV: first column is cell ID, remaining columns are gene counts.
2) metadata CSV: must contain a cell ID column and a label column.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autoencoder for scRNA-seq embeddings.")
    parser.add_argument("--counts", type=Path, required=True, help="Path to counts CSV.")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to metadata CSV.")
    parser.add_argument("--cell-col", type=str, default="cell", help="Cell ID column in metadata.")
    parser.add_argument(
        "--label-col",
        type=str,
        default="cell_ontology_class",
        help="Label column in metadata.",
    )
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--n-hvg", type=int, default=2000, help="Number of HVGs to keep.")
    parser.add_argument("--target-sum", type=float, default=1e4, help="Library size target.")
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent dimension.")
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="512,128",
        help="Comma-separated encoder hidden dims, e.g. 512,128.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument(
        "--loss",
        type=str,
        default="poisson",
        choices=["poisson", "mse", "mae", "huber"],
        help="Reconstruction loss. Poisson is recommended for non-negative expression values.",
    )
    parser.add_argument(
        "--huber-beta",
        type=float,
        default=1.0,
        help="Beta parameter for SmoothL1 (Huber) loss when --loss huber.",
    )
    parser.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction.")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience.")
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
        default="latent",
        choices=["latent", "input"],
        help="Which feature space to cluster on.",
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
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_hvg(log_expr: np.ndarray, gene_names: np.ndarray, n_hvg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select top genes by dispersion (variance / mean)."""
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

    return log_expr[:, top_idx], gene_names[top_idx], top_idx


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int) -> None:
        super().__init__()
        if input_dim < 2:
            raise ValueError("input_dim must be >= 2.")
        if latent_dim < 2:
            raise ValueError("latent_dim must be >= 2.")

        enc_layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            enc_layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: list[nn.Module] = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def get_device(mode: str) -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_loss(history_df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Autoencoder Training Curve")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_latent_2d(latent: np.ndarray, labels: np.ndarray, out_png: Path) -> None:
    plt.figure(figsize=(8, 6))
    unique_labels = pd.unique(labels.astype(str))
    for lab in unique_labels:
        idx = labels == lab
        plt.scatter(latent[idx, 0], latent[idx, 1], s=10, alpha=0.8, label=lab)
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("Autoencoder Latent Space (first 2 dims)")
    plt.legend(loc="best", markerscale=1.5, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_n = 0

    for (x,) in loader:
        x = x.to(device, non_blocking=True)
        x_hat, _ = model(x)
        loss = criterion(x_hat, x)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_n = x.size(0)
        total_loss += loss.item() * batch_n
        total_n += batch_n

    return total_loss / max(total_n, 1)


def build_criterion(args: argparse.Namespace) -> nn.Module:
    if args.loss == "mse":
        return nn.MSELoss()
    if args.loss == "mae":
        return nn.L1Loss()
    if args.loss == "huber":
        return nn.SmoothL1Loss(beta=args.huber_beta)
    if args.loss == "poisson":
        # Decoder output is interpreted as log-rate when log_input=True.
        return nn.PoissonNLLLoss(log_input=True, full=False)
    raise ValueError(f"Unsupported loss: {args.loss}")


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
    set_seed(args.seed)
    device = get_device(args.device)
    args.outdir.mkdir(parents=True, exist_ok=True)

    hidden_dims = [int(x.strip()) for x in args.hidden_dims.split(",") if x.strip()]
    if not hidden_dims:
        raise ValueError("At least one hidden dim is required.")

    counts_df = pd.read_csv(args.counts)
    if counts_df.shape[1] < 2:
        raise ValueError("Counts CSV must have one cell column + gene columns.")
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

    labels = metadata_df.loc[cell_ids, args.label_col].astype(str).values

    # Library-size normalization + log1p transform.
    libsize = expr.sum(axis=1, keepdims=True)
    libsize[libsize == 0] = 1.0
    expr_norm = (expr / libsize) * args.target_sum
    expr_log = np.log1p(expr_norm).astype(np.float32)

    # HVG selection is based on log-normalized expression.
    if args.n_hvg > 0:
        x_log_hvg, hvg_names, hvg_idx = select_hvg(expr_log, gene_names, args.n_hvg)
    else:
        hvg_idx = np.arange(gene_names.shape[0])
        x_log_hvg, hvg_names = expr_log, gene_names

    if args.loss == "poisson":
        # Poisson requires non-negative targets; use normalized expression directly.
        x_use = np.clip(expr_norm[:, hvg_idx], a_min=0.0, a_max=None).astype(np.float32)
    else:
        x_use = x_log_hvg.astype(np.float32)

    # Train/val split for robust stopping.
    idx_all = np.arange(x_use.shape[0])
    idx_train, idx_val = train_test_split(
        idx_all,
        test_size=args.val_frac,
        random_state=args.seed,
        shuffle=True,
        stratify=labels,
    )

    if args.loss == "poisson":
        # Keep non-negative scale for Poisson targets.
        x_model = x_use
        x_mean = np.zeros((1, x_use.shape[1]), dtype=np.float32)
        x_std = np.ones((1, x_use.shape[1]), dtype=np.float32)
    else:
        # Standardize using train statistics only for non-Poisson losses.
        x_mean = x_use[idx_train].mean(axis=0, keepdims=True)
        x_std = x_use[idx_train].std(axis=0, keepdims=True)
        x_std[x_std < 1e-6] = 1.0
        x_model = (x_use - x_mean) / x_std

    train_ds = TensorDataset(torch.from_numpy(x_model[idx_train]))
    val_ds = TensorDataset(torch.from_numpy(x_model[idx_val]))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = Autoencoder(input_dim=x_model.shape[1], hidden_dims=hidden_dims, latent_dim=args.latent_dim).to(device)
    criterion = build_criterion(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, criterion, None, device)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:03d}/{args.epochs} | train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
            break

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid model state.")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        x_tensor = torch.from_numpy(x_model).to(device)
        _, z_all = model(x_tensor)
        z_all = z_all.cpu().numpy()

    latent_cols = [f"z{i+1}" for i in range(z_all.shape[1])]
    latent_df = pd.DataFrame(z_all, columns=latent_cols)
    latent_df.insert(0, "cell", cell_ids.values)
    latent_df["label"] = labels
    latent_df.to_csv(args.outdir / "autoencoder_latent.csv", index=False)

    history_df = pd.DataFrame(history)
    history_df.to_csv(args.outdir / "train_history.csv", index=False)
    plot_loss(history_df, args.outdir / "train_curve.png")

    if z_all.shape[1] >= 2:
        plot_latent_2d(z_all[:, :2], labels, args.outdir / "latent_2d_plot.png")

    if args.cluster_on == "latent":
        cluster_emb = z_all
    else:
        cluster_emb = x_model

    metrics_rows, assignments = run_clustering(cluster_emb, labels.astype(str), args)
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(args.outdir / f"clustering_metrics_{args.cluster_on}.csv", index=False)
        for method, pred in assignments.items():
            out_df = pd.DataFrame(
                {
                    "cell": cell_ids.values,
                    "true_label": labels,
                    "cluster": pred,
                }
            )
            out_df.to_csv(args.outdir / f"cluster_assignments_{method}_{args.cluster_on}.csv", index=False)

    pd.Series(hvg_names, name="gene").to_csv(args.outdir / "hvg_genes.csv", index=False)
    if args.loss != "poisson":
        pd.DataFrame({"mean": x_mean.ravel(), "std": x_std.ravel()}, index=hvg_names).to_csv(
            args.outdir / "feature_scaler.csv"
        )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": x_model.shape[1],
            "hidden_dims": hidden_dims,
            "latent_dim": args.latent_dim,
            "loss": args.loss,
            "hvg_genes": hvg_names.tolist(),
            "x_mean": x_mean.astype(np.float32),
            "x_std": x_std.astype(np.float32),
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "seed": args.seed,
        },
        args.outdir / "autoencoder_model.pt",
    )

    print("\nDone.")
    print(f"Device: {device}")
    print(f"Cells: {expr.shape[0]}, genes: {expr.shape[1]}, HVGs used: {x_model.shape[1]}")
    print(f"Loss: {args.loss}")
    print(f"Best epoch: {best_epoch}, best val loss: {best_val:.6f}")
    print(f"Output directory: {args.outdir}")
    print("Saved files: autoencoder_latent.csv, train_history.csv, train_curve.png,")
    if args.loss == "poisson":
        print("             latent_2d_plot.png, hvg_genes.csv, autoencoder_model.pt")
    else:
        print("             latent_2d_plot.png, hvg_genes.csv, feature_scaler.csv, autoencoder_model.pt")
    if metrics_rows:
        print("             clustering_metrics_<space>.csv, cluster_assignments_*.csv")


if __name__ == "__main__":
    main()
