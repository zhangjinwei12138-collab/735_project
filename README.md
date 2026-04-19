# scRNA-seq Dimensionality Reduction + Clustering

This repo runs:
- `PCA + UMAP + clustering (GMM / DP)`
- `Autoencoder + clustering (GMM / DP)`

Main scripts:
- `run_pca_umap.py`
- `run_autoencoder.py`
- `run.sh`

## 1. Setup

```bash
cd /path/to/project_data
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Data Layout

By default, `run.sh` reads from `./data/`.
If `./data/` does not exist, it falls back to `./data copy/`.

Expected files:
- `brain_counts.csv`
- `brain_metadata.csv`

Expected schema:
- `brain_counts.csv`: first column is `cell id`, remaining columns are gene features.
- `brain_metadata.csv`: must include `cell` and `cell_ontology_class` columns.

## 3. Quick Run

```bash
bash run.sh
```

Outputs go to:
- `./outputs/pca_umap_hvg{N_HVG}_seed{SEED}/`
- `./outputs/ae_{LOSS}_hvg{N_HVG}_seed{SEED}/`

## 4. Useful Overrides

You can override defaults with env vars:

```bash
SEED=100 N_HVG=3000 CLUSTER_METHOD=both CLUSTER_ON=both bash run.sh
```

Run only PCA/UMAP:

```bash
RUN_PCA_UMAP=1 RUN_AE=0 bash run.sh
```

Run only Autoencoder:

```bash
RUN_PCA_UMAP=0 RUN_AE=1 AE_LOSS=poisson bash run.sh
```

Use a custom dataset path:

```bash
DATA_DIR=/absolute/path/to/data bash run.sh
```

## 5. Key Output Files

PCA/UMAP script outputs:
- `pca_embedding.csv`
- `umap_embedding.csv`
- `hvg_genes.csv`
- `pca_explained_variance_ratio.csv`
- `pca_plot.png`
- `umap_plot.png`
- `clustering_metrics_pca.csv` and/or `clustering_metrics_umap.csv`
- `cluster_assignments_{gmm|dp}_{pca|umap}.csv`

Autoencoder script outputs:
- `autoencoder_latent.csv`
- `train_history.csv`
- `train_curve.png`
- `latent_2d_plot.png`
- `hvg_genes.csv`
- `autoencoder_model.pt`
- `clustering_metrics_latent.csv` or `clustering_metrics_input.csv`
- `cluster_assignments_{gmm|dp}_{latent|input}.csv`

## 6. Reproducibility

- Set `SEED` in `run.sh` or via env var.
- UMAP with fixed seed is deterministic but may run single-threaded.
- Autoencoder results can still vary slightly across hardware/backends.
