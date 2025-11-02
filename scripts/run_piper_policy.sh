#!/usr/bin/env bash
# Helper to download the finetuned π₀.₅ Piper towel policy and launch the server.
#
# Usage:
#   ./scripts/run_piper_policy.sh
# Environment overrides (export or prefix on the command line):
#   HF_REPO_ID             Hugging Face repo that hosts the checkpoint       (default JessieLoki/pi05-piper-towel)
#   CHECKPOINT_STEP        Step directory to use (folder under the repo)     (default 999)
#   TARGET_DIR             Local path to place the checkpoint                (default checkpoints/pi05_piper_towel/piper_towel_full/${CHECKPOINT_STEP})
#   POLICY_CONFIG          Training config name to load                      (default pi05_piper_towel)
#   PORT                   Port for the policy server                        (default 8000)
#   INCLUDE_TRAIN_STATE    Set to 1 to also download optimizer state         (default 0)
#
# Prerequisites:
#   - `hf` CLI logged in with read access (run `hf auth login` once).
#   - `uv` available (the repo already uses it for training/inference).

set -euo pipefail

log() {
  echo "[piper-policy] $*" >&2
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

HF_REPO_ID=${HF_REPO_ID:-JessieLoki/pi05-piper-towel}
CHECKPOINT_STEP=${CHECKPOINT_STEP:-999}
TARGET_DIR=${TARGET_DIR:-"checkpoints/pi05_piper_towel/piper_towel_full/${CHECKPOINT_STEP}"}
POLICY_CONFIG=${POLICY_CONFIG:-pi05_piper_towel}
PORT=${PORT:-8000}
INCLUDE_TRAIN_STATE=${INCLUDE_TRAIN_STATE:-0}

if ! command -v hf >/dev/null 2>&1; then
  log "Missing Hugging Face CLI (hf). Install with: uv tool install huggingface_hub"
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  log "Missing uv. Install from https://docs.astral.sh/uv/"
  exit 1
fi

# Download checkpoint if not present.
if [[ ! -d "$TARGET_DIR/params" ]]; then
  log "Downloading checkpoint from $HF_REPO_ID to $TARGET_DIR"
  mkdir -p "$TARGET_DIR"
  DOWNLOAD_ARGS=(hf download "$HF_REPO_ID" --repo-type model --local-dir "$TARGET_DIR" --local-dir-use-symlinks False)
  DOWNLOAD_ARGS+=(--include "params/**" --include "assets/**")
  DOWNLOAD_ARGS+=(--exclude "_*")
  if [[ "$INCLUDE_TRAIN_STATE" != "1" ]]; then
    DOWNLOAD_ARGS+=(--exclude "train_state/**")
  fi
  log "Running: ${DOWNLOAD_ARGS[*]}"
  "${DOWNLOAD_ARGS[@]}"
else
  log "Found existing checkpoint at $TARGET_DIR; skipping download."
fi

if [[ ! -d "$TARGET_DIR/params" ]]; then
  log "Checkpoint download failed or missing params directory."
  exit 1
fi

log "Launching policy server on port ${PORT}"
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config="$POLICY_CONFIG" \
  --policy.dir="$TARGET_DIR" \
  --port="$PORT"
