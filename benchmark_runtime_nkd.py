#!/usr/bin/env python3
"""
Benchmark runtime scaling with N, K, and D for mixture clustering methods.

Outputs:
- runtime_benchmark_raw.csv
- runtime_benchmark_summary.csv
- runtime_vs_n.png
- runtime_vs_k.png
- runtime_vs_d.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture


def parse_int_list(s: str) -> list[int]:
    vals = [x.strip() for x in s.split(",") if x.strip()]
    out = [int(x) for x in vals]
    if not out:
        raise ValueError("List cannot be empty.")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark runtime scaling vs N/K/D.")
    parser.add_argument("--counts", type=Path, required=True, help="Path to counts CSV.")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--target-sum", type=float, default=1e4, help="Library size target.")
    parser.add_argument("--max-hvg", type=int, default=3000, help="Max HVGs used for benchmarking.")
    parser.add_argument("--n-values", type=str, default="500,1000,2000,3000", help="N sweep values.")
    parser.add_argument("--k-values", type=str, default="3,5,7,10,15", help="K sweep values.")
    parser.add_argument("--d-values", type=str, default="50,100,200,500,1000", help="D sweep values.")
    parser.add_argument("--n-fixed", type=int, default=2000, help="Fixed N for K/D sweeps.")
    parser.add_argument("--k-fixed", type=int, default=7, help="Fixed K for N/D sweeps.")
    parser.add_argument("--d-fixed", type=int, default=100, help="Fixed D for N/K sweeps.")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per setting.")
    parser.add_argument("--methods", type=str, default="gmm,dp", help="Methods: gmm,dp.")
    parser.add_argument(
        "--covariance-type",
        type=str,
        default="diag",
        choices=["full", "tied", "diag", "spherical"],
        help="Covariance type for mixture models.",
    )
    parser.add_argument(
        "--reg-covar",
        type=float,
        default=1e-5,
        help="Non-negative regularization added to the diagonal of covariance.",
    )
    parser.add_argument("--gmm-max-iter", type=int, default=200, help="Max iterations for mixture models.")
    parser.add_argument("--gmm-n-init", type=int, default=3, help="N init for mixture models.")
    parser.add_argument(
        "--dp-weight-concentration",
        type=float,
        default=1.0,
        help="Weight concentration prior for DP model.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def select_hvg(log_expr: np.ndarray, n_hvg: int) -> np.ndarray:
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
    return top_idx


def preprocess_counts(counts_path: Path, target_sum: float, max_hvg: int) -> np.ndarray:
    counts_df = pd.read_csv(counts_path)
    if counts_df.shape[1] < 2:
        raise ValueError("Counts CSV must have one cell id column + gene columns.")
    x = counts_df.iloc[:, 1:].to_numpy(dtype=np.float32)
    libsize = x.sum(axis=1, keepdims=True)
    libsize[libsize == 0] = 1.0
    x_norm = (x / libsize) * target_sum
    x_log = np.log1p(x_norm).astype(np.float32)

    hvg_idx = select_hvg(x_log, max_hvg)
    x_hvg = x_log[:, hvg_idx].astype(np.float32)

    # Standardize once for fair runtime comparisons.
    mu = x_hvg.mean(axis=0, keepdims=True)
    sd = x_hvg.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    x_std = (x_hvg - mu) / sd
    return x_std.astype(np.float32)


def fit_once(
    x: np.ndarray,
    method: str,
    k: int,
    args: argparse.Namespace,
    random_state: int,
) -> float:
    t0 = time.perf_counter()
    if method == "gmm":
        model = GaussianMixture(
            n_components=k,
            covariance_type=args.covariance_type,
            reg_covar=args.reg_covar,
            max_iter=args.gmm_max_iter,
            n_init=args.gmm_n_init,
            random_state=random_state,
        )
    elif method == "dp":
        model = BayesianGaussianMixture(
            n_components=k,
            covariance_type=args.covariance_type,
            reg_covar=args.reg_covar,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=args.dp_weight_concentration,
            max_iter=args.gmm_max_iter,
            n_init=max(1, args.gmm_n_init),
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    model.fit_predict(x)
    return time.perf_counter() - t0


def benchmark_axis(
    axis_name: str,
    values: list[int],
    x_all: np.ndarray,
    methods: list[str],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    n_total, d_total = x_all.shape

    n_fixed = min(args.n_fixed, n_total)
    d_fixed = min(args.d_fixed, d_total)

    for val in values:
        if axis_name == "N":
            n = min(val, n_total)
            d = d_fixed
            k = args.k_fixed
        elif axis_name == "D":
            n = n_fixed
            d = min(val, d_total)
            k = args.k_fixed
        elif axis_name == "K":
            n = n_fixed
            d = d_fixed
            k = val
        else:
            raise ValueError(f"Unknown axis: {axis_name}")

        for rep in range(args.repeats):
            idx = rng.choice(n_total, size=n, replace=False)
            x = x_all[idx, :d]
            for method in methods:
                try:
                    sec = fit_once(x, method, k, args, random_state=args.seed + rep)
                    rows.append(
                        {
                            "axis": axis_name,
                            "axis_value": val,
                            "method": method,
                            "repeat": rep,
                            "N": int(n),
                            "D": int(d),
                            "K": int(k),
                            "runtime_sec": float(sec),
                            "status": "ok",
                            "error": "",
                            "covariance_type": args.covariance_type,
                            "reg_covar": args.reg_covar,
                        }
                    )
                    print(
                        f"[{axis_name}] value={val} method={method} rep={rep+1}/{args.repeats} "
                        f"N={n} D={d} K={k} runtime={sec:.4f}s"
                    )
                except ValueError as e:
                    msg = str(e).replace("\n", " ")
                    rows.append(
                        {
                            "axis": axis_name,
                            "axis_value": val,
                            "method": method,
                            "repeat": rep,
                            "N": int(n),
                            "D": int(d),
                            "K": int(k),
                            "runtime_sec": float("nan"),
                            "status": "failed",
                            "error": msg,
                            "covariance_type": args.covariance_type,
                            "reg_covar": args.reg_covar,
                        }
                    )
                    print(
                        f"[{axis_name}] value={val} method={method} rep={rep+1}/{args.repeats} "
                        f"N={n} D={d} K={k} FAILED: {msg}"
                    )
    return rows


def plot_axis(summary_df: pd.DataFrame, axis_name: str, out_path: Path) -> None:
    df = summary_df[summary_df["axis"] == axis_name].sort_values(["method", "axis_value"])
    plt.figure(figsize=(7, 5))
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method].sort_values("axis_value")
        x = sub["axis_value"].to_numpy()
        y = sub["mean_runtime_sec"].to_numpy()
        s = sub["std_runtime_sec"].to_numpy()
        plt.plot(x, y, marker="o", label=method.upper())
        plt.fill_between(x, y - s, y + s, alpha=0.2)
    plt.xlabel(axis_name)
    plt.ylabel("Runtime (sec)")
    plt.title(f"Runtime vs {axis_name}")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    allowed = {"gmm", "dp"}
    if not methods or any(m not in allowed for m in methods):
        raise ValueError("methods must be comma-separated subset of: gmm,dp")

    n_values = parse_int_list(args.n_values)
    k_values = parse_int_list(args.k_values)
    d_values = parse_int_list(args.d_values)

    rng = np.random.default_rng(args.seed)
    x_all = preprocess_counts(args.counts, args.target_sum, args.max_hvg)
    print(f"Preprocessed matrix for benchmark: N={x_all.shape[0]}, D={x_all.shape[1]}")

    rows: list[dict[str, float | int | str]] = []
    rows.extend(benchmark_axis("N", n_values, x_all, methods, args, rng))
    rows.extend(benchmark_axis("K", k_values, x_all, methods, args, rng))
    rows.extend(benchmark_axis("D", d_values, x_all, methods, args, rng))

    raw_df = pd.DataFrame(rows)
    raw_path = args.outdir / "runtime_benchmark_raw.csv"
    raw_df.to_csv(raw_path, index=False)

    summary_df = (
        raw_df[raw_df["status"] == "ok"]
        .groupby(["axis", "axis_value", "method"], as_index=False)["runtime_sec"]
        .agg(mean_runtime_sec="mean", std_runtime_sec="std")
        .fillna(0.0)
    )
    summary_path = args.outdir / "runtime_benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    plot_axis(summary_df, "N", args.outdir / "runtime_vs_n.png")
    plot_axis(summary_df, "K", args.outdir / "runtime_vs_k.png")
    plot_axis(summary_df, "D", args.outdir / "runtime_vs_d.png")

    print(f"Saved: {raw_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {args.outdir / 'runtime_vs_n.png'}")
    print(f"Saved: {args.outdir / 'runtime_vs_k.png'}")
    print(f"Saved: {args.outdir / 'runtime_vs_d.png'}")


if __name__ == "__main__":
    main()
