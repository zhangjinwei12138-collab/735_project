# scRNA-seq Dimensionality Reduction + Clustering

Current workflow:
- `PCA + UMAP embedding`, but clustering is run on `PCA` space by default.
- `Autoencoder latent embedding + clustering (GMM / DP)`.
- `K-fold CV` for AE hyperparameter selection.

Main scripts:
- `run_pca_umap.py`
- `run_autoencoder.py`
- `run_autoencoder_cv.py`
- `plot_label_comparison.py`
- `benchmark_runtime_nkd.py`
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
pip install kaggle
chmod 600 ~/.kaggle/kaggle.json
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
- `brain_counts.csv`: first column is cell ID, remaining columns are gene features.
- `brain_metadata.csv`: must include `cell` and `cell_ontology_class`.

## 3. Quick Run

```bash
bash run.sh
```

Outputs go to:
- `./outputs/pca_umap_hvg{N_HVG}_seed{SEED}/`
- `./outputs/ae_{LOSS}_hvg{N_HVG}_seed{SEED}/`

## 4. Useful Overrides

Default clustering for PCA/UMAP script is `CLUSTER_ON=pca` (no UMAP clustering unless you set it).

```bash
SEED=123 N_HVG=3000 CLUSTER_METHOD=both CLUSTER_ON=pca bash run.sh
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
- `pca_explained_variance_ratio.png`
- `pca_plot.png`
- `umap_plot.png`
- `clustering_metrics_pca.csv` and/or `clustering_metrics_umap.csv` (depends on `--cluster-on`)
- `cluster_assignments_{gmm|dp}_{pca|umap}.csv` (depends on `--cluster-on`)

Autoencoder script outputs:
- `autoencoder_latent.csv`
- `train_history.csv`
- `train_curve.png`
- `latent_2d_plot.png`
- `ae_umap_embedding.csv`
- `ae_umap_plot.png`
- `hvg_genes.csv`
- `autoencoder_model.pt`
- `clustering_metrics_latent.csv` or `clustering_metrics_input.csv`
- `cluster_assignments_{gmm|dp}_{latent|input}.csv`

CV script outputs (`run_autoencoder_cv.py`):
- `cv_fold_results.csv`
- `cv_summary.csv`
- `cv_best_config.json`
- `run_best_command.sh`

## 6. Reproducibility Notes

- Set `SEED` in `run.sh` or as CLI arg.
- UMAP with fixed seed is deterministic but may run single-threaded.
- AE/GMM can still have small hardware/backend numerical differences.

## 7. Label Comparison Figure

Current plotting script defaults to panels:
- true label
- PCA + GMM
- PCA + DP
- AE + GMM
- AE + DP

```bash
python /proj/daiweilab/users/jinweizh/735/project_data/plot_label_comparison.py \
  --pca-dir /proj/daiweilab/users/jinweizh/735/project_data/pca_umap_out_hvg3000_var80_seed123 \
  --ae-dir /proj/daiweilab/users/jinweizh/735/project_data/best_model_run \
  --outdir /proj/daiweilab/users/jinweizh/735/project_data/results/figures \
  --viz-space umap
```

If you later want UMAP clustering panels back, add:

```bash
--include-umap-cluster
```

## 8. Runtime Benchmark vs N / K / D

```bash
python /proj/daiweilab/users/jinweizh/735/project_data/benchmark_runtime_nkd.py \
  --counts '/proj/daiweilab/users/jinweizh/735/project_data/data copy/brain_counts.csv' \
  --outdir '/proj/daiweilab/users/jinweizh/735/project_data/results/tables' \
  --max-hvg 3000 \
  --n-values 500,1000,2000,3000 \
  --k-values 3,5,7,10,15 \
  --d-values 50,100,200,500,1000 \
  --n-fixed 2000 \
  --k-fixed 7 \
  --d-fixed 100 \
  --repeats 3 \
  --methods gmm,dp \
  --covariance-type diag \
  --reg-covar 1e-5 \
  --seed 123
```

## 9. Reproducible Commands (Seed 123)

### 9.1 PCA/UMAP embedding + PCA clustering (PCA variance threshold 0.8)

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

### 9.2 Autoencoder CV (model selection)

```bash
python /proj/daiweilab/users/jinweizh/735/project_data/run_autoencoder_cv.py \
  --counts '/proj/daiweilab/users/jinweizh/735/project_data/data copy/brain_counts.csv' \
  --metadata '/proj/daiweilab/users/jinweizh/735/project_data/data copy/brain_metadata.csv' \
  --outdir /proj/daiweilab/users/jinweizh/735/project_data/outputs/ae_cv_hvg3000_seed123 \
  --n-hvg 3000 \
  --cv-folds 5 \
  --hidden-dims-grid '1024,512,128;1024,256;512,128' \
  --latent-dim-grid '16,32' \
  --loss-grid 'poisson,huber' \
  --lr-grid '5e-4,1e-3' \
  --weight-decay-grid '1e-4,1e-5' \
  --epochs 200 \
  --patience 30 \
  --batch-size 128 \
  --n-clusters 7 \
  --gmm-reg-covar 1e-6 \
  --gmm-retry-reg-covars '1e-5,1e-4,1e-3' \
  --select-metric ari \
  --seed 123 \
  --device auto
```

### 9.3 CV best model retrain (current best config)

Current best config from `outputs/ae_cv_hvg3000_seed123/cv_best_config.json`:
- `hidden_dims=512,128`
- `latent_dim=32`
- `loss=poisson`
- `lr=5e-4`
- `weight_decay=1e-4`

```bash
python /proj/daiweilab/users/jinweizh/735/project_data/run_autoencoder.py \
  --counts '/proj/daiweilab/users/jinweizh/735/project_data/data copy/brain_counts.csv' \
  --metadata '/proj/daiweilab/users/jinweizh/735/project_data/data copy/brain_metadata.csv' \
  --outdir /proj/daiweilab/users/jinweizh/735/project_data/best_model_run \
  --n-hvg 3000 \
  --loss poisson \
  --hidden-dims 512,128 \
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
