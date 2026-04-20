# Results Package

This folder contains publication/demo-ready outputs.

## figures/
- `label_comparison_umap.png`: true label vs 6 clustering methods (PCA/UMAP/AE × GMM/DP) on a shared UMAP view.
- `runtime_vs_n.png`: runtime scaling with sample size N.
- `runtime_vs_k.png`: runtime scaling with number of components K.
- `runtime_vs_d.png`: runtime scaling with feature dimension D.

## tables/
- `label_comparison_metrics_umap.csv`: ARI and displayed cluster counts for each panel.
- `clustering_metrics_pca.csv`: clustering metrics from PCA-space clustering.
- `clustering_metrics_umap.csv`: clustering metrics from UMAP-space clustering.
- `clustering_metrics_ae_latent.csv`: clustering metrics from AE latent-space clustering.
- `runtime_benchmark_summary.csv`: aggregated runtime means/stds.
- `runtime_benchmark_raw.csv`: per-run raw benchmark timings.
