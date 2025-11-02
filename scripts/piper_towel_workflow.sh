#!/usr/bin/env bash
# End-to-end helper for fine-tuning π₀.₅ on the Piper towel dataset.
# It:
#   1. Loads login tokens from a secrets env file (if present).
#   2. Converts the raw Piper dataset on Hugging Face into the LeRobot layout.
#   3. Computes normalization statistics for the pi05_piper_towel config.
#   4. Launches training with optional overrides.
#
# Tweak behaviour via environment variables before running, e.g.:
#   SOURCE_REPO_ID=ETHRC/piper_towel_v0_with_rewards \
#   TARGET_REPO_ID=local/piper_towel_v0_converted \
#   EXP_NAME=piper_towel_full \
#   ./scripts/piper_towel_workflow.sh

set -euo pipefail

log() {
  echo "[piper-workflow] $*" >&2
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

SECRETS_FILE=${SECRETS_FILE:-secrets.env}
if [[ -f "$SECRETS_FILE" ]]; then
  log "Loading secrets from $SECRETS_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$SECRETS_FILE"
  set +a
else
  log "Secrets file $SECRETS_FILE not found; continuing without it."
fi

SOURCE_REPO_ID=${SOURCE_REPO_ID:-ETHRC/piper_towel_v0_with_rewards}
TARGET_REPO_ID=${TARGET_REPO_ID:-local/piper_towel_v0_converted}
SPLIT=${SPLIT:-train}
DOWNLOAD_VIDEOS=${DOWNLOAD_VIDEOS:-true}
MAX_EPISODES=${MAX_EPISODES:-}
FORCE_RECONVERT=${FORCE_RECONVERT:-0}
CONFIG_NAME=${CONFIG_NAME:-pi05_piper_towel}
EXP_NAME=${EXP_NAME:-piper_towel_full}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-}
BATCH_SIZE=${BATCH_SIZE:-}
XLA_MEM_FRACTION=${XLA_MEM_FRACTION:-0.9}

CACHE_ROOT=${HF_LEROBOT_HOME:-"$HOME/.cache/huggingface/lerobot"}
DATASET_DIR="${CACHE_ROOT}/${TARGET_REPO_ID}"
INFO_JSON="${DATASET_DIR}/meta/info.json"

DOWNLOAD_FLAG="True"
if [[ "${DOWNLOAD_VIDEOS,,}" == "false" ]]; then
  DOWNLOAD_FLAG="False"
fi

run_cmd() {
  log "⇒ $*"
  "$@"
}

if [[ "$FORCE_RECONVERT" == "1" || ! -f "$INFO_JSON" ]]; then
  log "Converting dataset ${SOURCE_REPO_ID} → ${TARGET_REPO_ID}"
  CONVERT_CMD=(uv run examples/piper_examples/convert_piper_data_to_lerobot.py
               --source-repo-id "$SOURCE_REPO_ID"
               --target-repo-id "$TARGET_REPO_ID"
               --split "$SPLIT"
               --download-videos "$DOWNLOAD_FLAG"
               --overwrite)
  if [[ -n "$MAX_EPISODES" ]]; then
    CONVERT_CMD+=(--max-episodes "$MAX_EPISODES")
  fi
  run_cmd "${CONVERT_CMD[@]}"
else
  log "Dataset already cached at ${DATASET_DIR}; skipping conversion."
fi

run_cmd uv run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"

TRAIN_CMD=(uv run scripts/train.py "$CONFIG_NAME" --exp-name "$EXP_NAME" --overwrite)
if [[ -n "$NUM_TRAIN_STEPS" ]]; then
  TRAIN_CMD+=(--num-train-steps "$NUM_TRAIN_STEPS")
fi
if [[ -n "$BATCH_SIZE" ]]; then
  TRAIN_CMD+=(--batch-size "$BATCH_SIZE")
fi

log "⇒ XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_MEM_FRACTION} ${TRAIN_CMD[*]}"
XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_MEM_FRACTION" "${TRAIN_CMD[@]}"

log "Done. Checkpoints: checkpoints/${CONFIG_NAME}/${EXP_NAME}"
