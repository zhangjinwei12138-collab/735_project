#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- Config (override with env vars if needed) ----------
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
if [[ ! -d "${DATA_DIR}" && -d "${SCRIPT_DIR}/data copy" ]]; then
  DATA_DIR="${SCRIPT_DIR}/data copy"
fi

COUNTS="${COUNTS:-${DATA_DIR}/brain_counts.csv}"
META="${META:-${DATA_DIR}/brain_metadata.csv}"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/outputs}"

SEED="${SEED:-42}"
N_HVG="${N_HVG:-3000}"
N_PCS="${N_PCS:-30}"
N_CLUSTERS="${N_CLUSTERS:-7}"
DP_MAX_COMPONENTS="${DP_MAX_COMPONENTS:-20}"
CLUSTER_METHOD="${CLUSTER_METHOD:-both}"      # none|gmm|dp|both
CLUSTER_ON="${CLUSTER_ON:-pca}"               # pca|umap|both (for run_pca_umap.py)
RUN_PCA_UMAP="${RUN_PCA_UMAP:-1}"             # 1 or 0
RUN_AE="${RUN_AE:-1}"                         # 1 or 0

# Autoencoder config
AE_LOSS="${AE_LOSS:-poisson}"                 # poisson|mse|mae|huber
AE_HIDDEN_DIMS="${AE_HIDDEN_DIMS:-1024,512,128}"
AE_LATENT_DIM="${AE_LATENT_DIM:-32}"
AE_EPOCHS="${AE_EPOCHS:-200}"
AE_PATIENCE="${AE_PATIENCE:-30}"
AE_BATCH_SIZE="${AE_BATCH_SIZE:-128}"
AE_LR="${AE_LR:-5e-4}"
AE_WEIGHT_DECAY="${AE_WEIGHT_DECAY:-1e-4}"
AE_CLUSTER_ON="${AE_CLUSTER_ON:-latent}"      # latent|input
AE_DEVICE="${AE_DEVICE:-auto}"                # auto|cpu|cuda

mkdir -p "${OUT_ROOT}"

if [[ ! -f "${COUNTS}" ]]; then
  echo "ERROR: counts file not found: ${COUNTS}" >&2
  exit 1
fi
if [[ ! -f "${META}" ]]; then
  echo "ERROR: metadata file not found: ${META}" >&2
  exit 1
fi

echo "Project dir: ${SCRIPT_DIR}"
echo "Data dir:    ${DATA_DIR}"
echo "Output dir:  ${OUT_ROOT}"
echo "Seed:        ${SEED}"

if [[ "${RUN_PCA_UMAP}" == "1" ]]; then
  PCA_OUT="${OUT_ROOT}/pca_umap_hvg${N_HVG}_seed${SEED}"
  mkdir -p "${PCA_OUT}"

  declare -a SPACES
  if [[ "${CLUSTER_ON}" == "both" ]]; then
    SPACES=("pca" "umap")
  else
    SPACES=("${CLUSTER_ON}")
  fi

  for SPACE in "${SPACES[@]}"; do
    echo "[PCA/UMAP] cluster_on=${SPACE}"
    python "${SCRIPT_DIR}/run_pca_umap.py" \
      --counts "${COUNTS}" \
      --metadata "${META}" \
      --outdir "${PCA_OUT}" \
      --n-hvg "${N_HVG}" \
      --n-pcs "${N_PCS}" \
      --cluster-method "${CLUSTER_METHOD}" \
      --cluster-on "${SPACE}" \
      --n-clusters "${N_CLUSTERS}" \
      --dp-max-components "${DP_MAX_COMPONENTS}" \
      --seed "${SEED}"
  done
fi

if [[ "${RUN_AE}" == "1" ]]; then
  AE_OUT="${OUT_ROOT}/ae_${AE_LOSS}_hvg${N_HVG}_seed${SEED}"
  mkdir -p "${AE_OUT}"

  echo "[Autoencoder] loss=${AE_LOSS}, cluster_on=${AE_CLUSTER_ON}"
  python "${SCRIPT_DIR}/run_autoencoder.py" \
    --counts "${COUNTS}" \
    --metadata "${META}" \
    --outdir "${AE_OUT}" \
    --n-hvg "${N_HVG}" \
    --loss "${AE_LOSS}" \
    --hidden-dims "${AE_HIDDEN_DIMS}" \
    --latent-dim "${AE_LATENT_DIM}" \
    --epochs "${AE_EPOCHS}" \
    --patience "${AE_PATIENCE}" \
    --batch-size "${AE_BATCH_SIZE}" \
    --lr "${AE_LR}" \
    --weight-decay "${AE_WEIGHT_DECAY}" \
    --cluster-method "${CLUSTER_METHOD}" \
    --cluster-on "${AE_CLUSTER_ON}" \
    --n-clusters "${N_CLUSTERS}" \
    --dp-max-components "${DP_MAX_COMPONENTS}" \
    --seed "${SEED}" \
    --device "${AE_DEVICE}"
fi

echo "Done."

python "${BASE_DIR}/run_pca_umap.py" \
  --counts "${COUNTS}" \
  --metadata "${META}" \
  --outdir "${BASE_DIR}/pca_umap_out_hvg3000_seed${SEED}" \
  --n-hvg 3000 \
  --n-pcs 30 \
  --cluster-method both \
  --cluster-on umap \
  --n-clusters 7 \
  --seed "${SEED}"
