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

Dataset link:
- https://www.kaggle.com/datasets/aayush9753/singlecell-rnaseq-data-from-mouse-brain

### 2.1 Download + Unzip (Kaggle CLI)

```bash
# one-time setup (if needed)
pip install kaggle

# put kaggle.json in ~/.kaggle/kaggle.json and set permission
chmod 600 ~/.kaggle/kaggle.json

# download and unzip directly into ./data
mkdir -p ./data
kaggle datasets download -d aayush9753/singlecell-rnaseq-data-from-mouse-brain -p ./data --unzip
```

### 2.2 If you downloaded a zip manually

```bash
mkdir -p ./data
unzip /path/to/singlecell-rnaseq-data-from-mouse-brain.zip -d ./data
```

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

## 7. True-vs-Cluster Comparison Figure

Generate one figure with:
- true label
- PCA+GMM, PCA+DP
- UMAP+GMM, UMAP+DP
- AE+GMM, AE+DP

```bash
python plot_label_comparison.py \
  --pca-dir ./pca_umap_out_hvg3000_var80_seed123 \
  --ae-dir ./ae_out_run1 \
  --outdir ./outputs/fig_compare \
  --viz-space umap
```

Outputs:
- `label_comparison_umap.png`
- `label_comparison_metrics_umap.csv`

## 8. Runtime Benchmark vs N / K / D

Benchmark computational runtime for `GMM` and `DP` as N/K/D vary:

```bash
python benchmark_runtime_nkd.py \
  --counts "./data/brain_counts.csv" \
  --outdir "./outputs/benchmark_nkd" \
  --max-hvg 3000 \
  --n-values 500,1000,2000,3000 \
  --k-values 3,5,7,10,15 \
  --d-values 50,100,200,500,1000 \
  --n-fixed 2000 \
  --k-fixed 7 \
  --d-fixed 100 \
  --repeats 3 \
  --methods gmm,dp
```

Outputs:
- `runtime_benchmark_raw.csv`
- `runtime_benchmark_summary.csv`
- `runtime_vs_n.png`
- `runtime_vs_k.png`
- `runtime_vs_d.png`

## 9. Reproducible Commands (Seed 123)

Autoencoder:

```bash
python /proj/daiweilab/users/jinweizh/735/project_data/run_autoencoder.py \
  --counts '/proj/daiweilab/users/jinweizh/735/project_data/data copy/brain_counts.csv' \
  --metadata '/proj/daiweilab/users/jinweizh/735/project_data/data copy/brain_metadata.csv' \
  --outdir /proj/daiweilab/users/jinweizh/735/project_data/ae_out_run1 \
  --n-hvg 3000 \
  --loss poisson \
  --hidden-dims 1024,512,128 \
  --latent-dim 32 \
  --epochs 200 \
  --patience 30 \
  --batch-size 128 \
  --lr 5e-4 \
  --weight-decay 1e-4 \
  --cluster-method both \
  --cluster-on latent \
  --n-clusters 7 \
  --dp-max-components 20 \
  --seed 123 \
  --device auto
```

PCA/UMAP (cluster on PCA, 80% cumulative explained variance target):

```bash
python /proj/daiweilab/users/jinweizh/735/project_data/run_pca_umap.py \
  --counts '/proj/daiweilab/users/jinweizh/735/project_data/data copy/brain_counts.csv' \
  --metadata '/proj/daiweilab/users/jinweizh/735/project_data/data copy/brain_metadata.csv' \
  --outdir /proj/daiweilab/users/jinweizh/735/project_data/pca_umap_out_hvg3000_var80_seed123 \
  --n-hvg 3000 \
  --pca-var-threshold 0.8 \
  --cluster-method both \
  --cluster-on pca \
  --n-clusters 7 \
  --seed 123
```
