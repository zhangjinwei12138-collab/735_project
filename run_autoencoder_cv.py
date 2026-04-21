#!/usr/bin/env python3
"""
Cross-validation model selection for autoencoder-based scRNA embedding.

This script:
1) loads counts + metadata
2) preprocesses (library normalize + log1p + HVG)
3) runs K-fold CV over hyperparameter grids
4) selects best config by mean ARI (default) or mean val_loss
5) saves summaries and a runnable command for run_autoencoder.py
"""

from __future__ import annotations

import argparse
import json
import random
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K-fold CV for run_autoencoder hyperparameter selection.")
    parser.add_argument("--counts", type=Path, required=True, help="Path to counts CSV.")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to metadata CSV.")
    parser.add_argument("--cell-col", type=str, default="cell", help="Cell ID column in metadata.")
    parser.add_argument("--label-col", type=str, default="cell_ontology_class", help="Label column in metadata.")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory.")

    parser.add_argument("--n-hvg", type=int, default=3000, help="Number of HVGs.")
    parser.add_argument("--target-sum", type=float, default=1e4, help="Library size target.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument(
        "--hidden-dims-grid",
        type=str,
        default="1024,512,128;1024,256;512,128",
        help="Semicolon-separated hidden-dims candidates. Example: 1024,512,128;1024,256",
    )
    parser.add_argument("--latent-dim-grid", type=str, default="16,32", help="Comma-separated latent dims.")
    parser.add_argument("--loss-grid", type=str, default="poisson,huber", help="Comma-separated losses.")
    parser.add_argument("--lr-grid", type=str, default="5e-4,1e-3", help="Comma-separated learning rates.")
    parser.add_argument(
        "--weight-decay-grid",
        type=str,
        default="1e-4,1e-5",
        help="Comma-separated weight decay values.",
    )

    parser.add_argument("--epochs", type=int, default=200, help="Training epochs per fold.")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--huber-beta", type=float, default=1.0, help="Huber beta when loss=huber.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--n-clusters", type=int, default=7, help="GMM clusters for ARI evaluation.")
    parser.add_argument(
        "--covariance-type",
        type=str,
        default="diag",
        choices=["full", "tied", "diag", "spherical"],
        help="Covariance type for GMM.",
    )
    parser.add_argument("--gmm-max-iter", type=int, default=200, help="GMM max iterations.")
    parser.add_argument("--gmm-n-init", type=int, default=5, help="GMM n_init.")
    parser.add_argument(
        "--gmm-reg-covar",
        type=float,
        default=1e-6,
        help="Initial reg_covar for GMM numerical stability.",
    )
    parser.add_argument(
        "--gmm-retry-reg-covars",
        type=str,
        default="1e-5,1e-4,1e-3",
        help="Comma-separated fallback reg_covar values if GMM fails.",
    )

    parser.add_argument(
        "--select-metric",
        type=str,
        default="ari",
        choices=["ari", "val_loss"],
        help="Metric for selecting best config.",
    )
    parser.add_argument(
        "--max-combos",
        type=int,
        default=0,
        help="If >0, run only first N hyperparameter combinations.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(mode: str) -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_hidden_grid(s: str) -> list[list[int]]:
    items = [x.strip() for x in s.split(";") if x.strip()]
    out: list[list[int]] = []
    for item in items:
        dims = [int(v.strip()) for v in item.split(",") if v.strip()]
        if not dims:
            continue
        out.append(dims)
    if not out:
        raise ValueError("hidden-dims-grid is empty.")
    return out


def parse_int_list(s: str) -> list[int]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("int grid is empty.")
    return vals


def parse_float_list(s: str) -> list[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("float grid is empty.")
    return vals


def parse_str_list(s: str) -> list[str]:
    vals = [x.strip() for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("string grid is empty.")
    return vals


def safe_mean(xs: list[float]) -> float:
    arr = np.asarray(xs, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def safe_std(xs: list[float]) -> float:
    arr = np.asarray(xs, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanstd(arr))


def select_hvg(log_expr: np.ndarray, gene_names: np.ndarray, n_hvg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = log_expr.mean(axis=0)
    vars_ = log_expr.var(axis=0)
    disp = vars_ / (means + 1e-8)
    valid = np.isfinite(disp) & (means > 0)
    valid_idx = np.where(valid)[0]
    if valid_idx.size == 0:
        raise ValueError("No valid genes for HVG selection.")
    keep = min(n_hvg, valid_idx.size)
    top_local = np.argpartition(disp[valid], -keep)[-keep:]
    top_idx = valid_idx[top_local]
    top_idx = top_idx[np.argsort(disp[top_idx])[::-1]]
    return log_expr[:, top_idx], gene_names[top_idx], top_idx


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int) -> None:
        super().__init__()
        enc: list[nn.Module] = []
        d = input_dim
        for h in hidden_dims:
            enc.extend([nn.Linear(d, h), nn.ReLU()])
            d = h
        enc.append(nn.Linear(d, latent_dim))
        self.encoder = nn.Sequential(*enc)

        dec: list[nn.Module] = []
        d = latent_dim
        for h in reversed(hidden_dims):
            dec.extend([nn.Linear(d, h), nn.ReLU()])
            d = h
        dec.append(nn.Linear(d, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def build_criterion(loss_name: str, huber_beta: float) -> nn.Module:
    if loss_name == "poisson":
        return nn.PoissonNLLLoss(log_input=True, full=False)
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "mae":
        return nn.L1Loss()
    if loss_name == "huber":
        return nn.SmoothL1Loss(beta=huber_beta)
    raise ValueError(f"Unsupported loss: {loss_name}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_n = 0
    for (x,) in loader:
        x = x.to(device, non_blocking=True)
        x_hat, _ = model(x)
        loss = criterion(x_hat, x)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        total_n += x.size(0)
    return total_loss / max(total_n, 1)


def preprocess_inputs(
    counts_path: Path,
    meta_path: Path,
    cell_col: str,
    label_col: str,
    n_hvg: int,
    target_sum: float,
) -> dict[str, object]:
    counts_df = pd.read_csv(counts_path)
    meta_df = pd.read_csv(meta_path)
    if counts_df.shape[1] < 2:
        raise ValueError("Counts CSV must have one cell column + gene columns.")
    for col in [cell_col, label_col]:
        if col not in meta_df.columns:
            raise ValueError(f"Column '{col}' not found in metadata.")

    cell_ids = counts_df.iloc[:, 0].astype(str).values
    gene_names = counts_df.columns[1:].to_numpy()
    expr = counts_df.iloc[:, 1:].to_numpy(dtype=np.float32)

    meta_df[cell_col] = meta_df[cell_col].astype(str)
    meta_df = meta_df.set_index(cell_col)
    if not pd.Index(cell_ids).isin(meta_df.index).all():
        raise ValueError("Some cells in counts are missing in metadata.")
    labels = meta_df.loc[cell_ids, label_col].astype(str).values

    libsize = expr.sum(axis=1, keepdims=True)
    libsize[libsize == 0] = 1.0
    expr_norm = (expr / libsize) * target_sum
    expr_log = np.log1p(expr_norm).astype(np.float32)

    x_log_hvg, hvg_names, hvg_idx = select_hvg(expr_log, gene_names, n_hvg)
    x_norm_hvg = np.clip(expr_norm[:, hvg_idx], a_min=0.0, a_max=None).astype(np.float32)

    return {
        "cell_ids": cell_ids,
        "labels": labels,
        "hvg_names": hvg_names,
        "x_log_hvg": x_log_hvg.astype(np.float32),
        "x_norm_hvg": x_norm_hvg.astype(np.float32),
    }


def prepare_fold_matrix(
    x_log_hvg: np.ndarray,
    x_norm_hvg: np.ndarray,
    train_idx: np.ndarray,
    loss_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if loss_name == "poisson":
        x_model = x_norm_hvg
        mean = np.zeros((1, x_model.shape[1]), dtype=np.float32)
        std = np.ones((1, x_model.shape[1]), dtype=np.float32)
    else:
        x_model = x_log_hvg
        mean = x_model[train_idx].mean(axis=0, keepdims=True)
        std = x_model[train_idx].std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        x_model = (x_model - mean) / std
    return x_model.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def train_one_fold(
    x_model: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    hidden_dims: list[int],
    latent_dim: int,
    loss_name: str,
    lr: float,
    weight_decay: float,
    args: argparse.Namespace,
    device: torch.device,
    fold_seed: int,
) -> tuple[float, np.ndarray]:
    set_seed(fold_seed)
    train_ds = TensorDataset(torch.from_numpy(x_model[train_idx]))
    val_ds = TensorDataset(torch.from_numpy(x_model[val_idx]))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = Autoencoder(input_dim=x_model.shape[1], hidden_dims=hidden_dims, latent_dim=latent_dim).to(device)
    criterion = build_criterion(loss_name, args.huber_beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0

    for _ in range(args.epochs):
        _ = run_epoch(model, train_loader, criterion, device, optimizer)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, criterion, device, optimizer=None)
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("No valid model state found.")
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        z_val = model.encoder(torch.from_numpy(x_model[val_idx]).to(device)).cpu().numpy()
    return float(best_val), z_val


def fit_gmm_with_retry(
    z_val: np.ndarray,
    n_clusters: int,
    covariance_type: str,
    max_iter: int,
    n_init: int,
    random_state: int,
    reg_covars: list[float],
) -> tuple[np.ndarray, str, float]:
    errors: list[str] = []
    cov_types = [covariance_type]
    if covariance_type != "diag":
        cov_types.append("diag")

    seen: set[tuple[str, float]] = set()
    for cov in cov_types:
        for reg in reg_covars:
            key = (cov, float(reg))
            if key in seen:
                continue
            seen.add(key)
            try:
                gmm = GaussianMixture(
                    n_components=n_clusters,
                    covariance_type=cov,
                    max_iter=max_iter,
                    n_init=n_init,
                    reg_covar=float(reg),
                    random_state=random_state,
                )
                pred = gmm.fit_predict(z_val)
                return pred, cov, float(reg)
            except ValueError as e:
                errors.append(f"cov={cov}, reg_covar={reg}: {e}")
    raise ValueError(" | ".join(errors))


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    set_seed(args.seed)

    hidden_grid = parse_hidden_grid(args.hidden_dims_grid)
    latent_grid = parse_int_list(args.latent_dim_grid)
    loss_grid = parse_str_list(args.loss_grid)
    lr_grid = parse_float_list(args.lr_grid)
    wd_grid = parse_float_list(args.weight_decay_grid)
    retry_regs = [args.gmm_reg_covar] + parse_float_list(args.gmm_retry_reg_covars)

    combos = list(product(hidden_grid, latent_grid, loss_grid, lr_grid, wd_grid))
    if args.max_combos > 0:
        combos = combos[: args.max_combos]

    data = preprocess_inputs(
        counts_path=args.counts,
        meta_path=args.metadata,
        cell_col=args.cell_col,
        label_col=args.label_col,
        n_hvg=args.n_hvg,
        target_sum=args.target_sum,
    )
    labels = data["labels"]
    x_log_hvg = data["x_log_hvg"]
    x_norm_hvg = data["x_norm_hvg"]

    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    folds = list(skf.split(np.arange(len(labels)), labels))

    fold_rows: list[dict[str, object]] = []
    combo_rows: list[dict[str, object]] = []

    for combo_id, (hidden_dims, latent_dim, loss_name, lr, weight_decay) in enumerate(combos, start=1):
        ari_scores: list[float] = []
        val_losses: list[float] = []

        print(
            f"[combo {combo_id}/{len(combos)}] hidden={hidden_dims} latent={latent_dim} "
            f"loss={loss_name} lr={lr} wd={weight_decay}"
        )

        for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
            x_model, _, _ = prepare_fold_matrix(x_log_hvg, x_norm_hvg, train_idx, loss_name)
            val_loss, z_val = train_one_fold(
                x_model=x_model,
                train_idx=train_idx,
                val_idx=val_idx,
                hidden_dims=hidden_dims,
                latent_dim=latent_dim,
                loss_name=loss_name,
                lr=lr,
                weight_decay=weight_decay,
                args=args,
                device=device,
                fold_seed=args.seed + fold_id,
            )

            gmm_status = "ok"
            gmm_covariance_used = args.covariance_type
            gmm_reg_covar_used = float("nan")
            gmm_error = ""
            try:
                pred, gmm_covariance_used, gmm_reg_covar_used = fit_gmm_with_retry(
                    z_val=z_val,
                    n_clusters=args.n_clusters,
                    covariance_type=args.covariance_type,
                    max_iter=args.gmm_max_iter,
                    n_init=args.gmm_n_init,
                    random_state=args.seed + fold_id,
                    reg_covars=retry_regs,
                )
                ari = float(adjusted_rand_score(labels[val_idx], pred))
            except ValueError as e:
                gmm_status = "failed"
                gmm_error = str(e)
                ari = float("nan")
                print(
                    f"  fold {fold_id}/{args.cv_folds}: GMM failed after retries; "
                    f"set ari=nan. {e}"
                )

            ari_scores.append(ari)
            val_losses.append(val_loss)
            fold_rows.append(
                {
                    "combo_id": combo_id,
                    "fold": fold_id,
                    "hidden_dims": ",".join(map(str, hidden_dims)),
                    "latent_dim": latent_dim,
                    "loss": loss_name,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "val_loss": val_loss,
                    "ari": ari,
                    "gmm_status": gmm_status,
                    "gmm_covariance_used": gmm_covariance_used,
                    "gmm_reg_covar_used": gmm_reg_covar_used,
                    "gmm_error": gmm_error,
                }
            )
            if np.isnan(ari):
                print(f"  fold {fold_id}/{args.cv_folds}: val_loss={val_loss:.6f}, ari=nan")
            else:
                print(f"  fold {fold_id}/{args.cv_folds}: val_loss={val_loss:.6f}, ari={ari:.4f}")

        combo_rows.append(
            {
                "combo_id": combo_id,
                "hidden_dims": ",".join(map(str, hidden_dims)),
                "latent_dim": latent_dim,
                "loss": loss_name,
                "lr": lr,
                "weight_decay": weight_decay,
                "mean_val_loss": safe_mean(val_losses),
                "std_val_loss": safe_std(val_losses),
                "mean_ari": safe_mean(ari_scores),
                "std_ari": safe_std(ari_scores),
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    combo_df = pd.DataFrame(combo_rows)

    if args.select_metric == "ari":
        valid = combo_df["mean_ari"].notna()
        if not valid.any():
            raise RuntimeError(
                "All combos have NaN ARI (GMM failed in all folds). "
                "Try larger --gmm-reg-covar and --gmm-retry-reg-covars, or reduce --n-clusters."
            )
        best_idx = combo_df.loc[valid, "mean_ari"].idxmax()
    else:
        best_idx = combo_df["mean_val_loss"].idxmin()
    best_row = combo_df.loc[best_idx].to_dict()

    fold_df.to_csv(args.outdir / "cv_fold_results.csv", index=False)
    combo_df.sort_values("mean_ari", ascending=False).to_csv(args.outdir / "cv_summary.csv", index=False)

    with open(args.outdir / "cv_best_config.json", "w", encoding="utf-8") as f:
        json.dump(best_row, f, indent=2)

    best_cmd = (
        f"python run_autoencoder.py --counts '{args.counts}' --metadata '{args.metadata}' "
        f"--outdir '{args.outdir / 'best_model_run'}' --n-hvg {args.n_hvg} "
        f"--loss {best_row['loss']} --hidden-dims {best_row['hidden_dims']} "
        f"--latent-dim {int(best_row['latent_dim'])} --epochs {args.epochs} "
        f"--patience {args.patience} --batch-size {args.batch_size} "
        f"--lr {best_row['lr']} --weight-decay {best_row['weight_decay']} "
        f"--cluster-method both --cluster-on latent --n-clusters {args.n_clusters} "
        f"--seed {args.seed} --device {args.device}"
    )
    with open(args.outdir / "run_best_command.sh", "w", encoding="utf-8") as f:
        f.write(best_cmd + "\n")

    print("\nCV done.")
    print(f"Device: {device}")
    print(f"Saved: {args.outdir / 'cv_fold_results.csv'}")
    print(f"Saved: {args.outdir / 'cv_summary.csv'}")
    print(f"Saved: {args.outdir / 'cv_best_config.json'}")
    print(f"Saved: {args.outdir / 'run_best_command.sh'}")
    print(f"Best by {args.select_metric}:")
    print(best_row)


if __name__ == "__main__":
    main()
