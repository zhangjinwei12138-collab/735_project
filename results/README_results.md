# Results Package

This folder contains publication/demo-ready outputs.
All files are reproducible from scripts in project root.

## figures/
- `label_comparison_umap.png`: true label vs clustered labels on a shared UMAP view (current panels: true, PCA+GMM, PCA+DP, AE+GMM, AE+DP).
- `label_comparison_metrics_umap.csv`: ARI and displayed cluster counts for panels in `label_comparison_umap.png`.
- `runtime_vs_n.png`: runtime scaling with sample size N.
- `runtime_vs_k.png`: runtime scaling with number of components K.
- `runtime_vs_d.png`: runtime scaling with feature dimension D.

## tables/
- `runtime_benchmark_summary.csv`: aggregated runtime means/stds.
- `runtime_benchmark_raw.csv`: per-run raw benchmark timings.

## How These Figures Were Generated

### Label comparison figure
- Script: `plot_label_comparison.py`
- Inputs:
  - PCA/UMAP directory from `run_pca_umap.py`
  - AE directory from `run_autoencoder.py`
- Note: current figure was generated without UMAP clustering panels (default mode of script).

### Runtime benchmark figures
- Script: `benchmark_runtime_nkd.py`
- It writes `runtime_vs_{n,k,d}.png` and benchmark tables to the same output directory.
