#!/usr/bin/env bash
set -euo pipefail
trap 'echo "✖ Upload interrupted."; exit 1' INT

HERE="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${HERE}/.." && pwd)"

# Load envs (prefer .env.upload if available)
if [[ -f "${REPO_ROOT}/.env.upload" ]]; then
  set -a
  source "${REPO_ROOT}/.env.upload"
  set +a
elif [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  source "${REPO_ROOT}/.env"
  set +a
else
  echo "✖ No .env or .env.upload found in repo root." >&2
  exit 1
fi

# Check mandatory vars
: "${UPLOAD_USER:?Set UPLOAD_USER in .env.upload or .env}"
: "${UPLOAD_HOST:?Set UPLOAD_HOST in .env.upload or .env}"
: "${UPLOAD_PATH:?Set UPLOAD_PATH in .env.upload or .env}"

# Input check
LOCAL_SRC="${1:-}"
if [[ -z "$LOCAL_SRC" ]]; then
  echo "Usage: ./data_preprocessing/upload_datasets.sh <path_to_file_or_folder>" >&2
  exit 1
fi
if [[ ! -e "$LOCAL_SRC" ]]; then
  echo "✖ Path does not exist: $LOCAL_SRC" >&2
  exit 1
fi

# Absolute path + remote folder
LOCAL_ABS="$(cd "$(dirname "$LOCAL_SRC")" && pwd)/$(basename "$LOCAL_SRC")"
BASENAME="$(basename "$LOCAL_SRC")"
REMOTE_TARGET="${UPLOAD_PATH}/${BASENAME}"

echo "📦 Uploading:"
echo "   ${LOCAL_ABS}"
echo "🛰️  → ${UPLOAD_USER}@${UPLOAD_HOST}:${REMOTE_TARGET}"

# Create target directory
ssh "${UPLOAD_USER}@${UPLOAD_HOST}" "mkdir -p '${REMOTE_TARGET}'"

# Rsync (follow symlinks, resume if interrupted)
rsync -av --partial --progress -L -e ssh "${LOCAL_ABS}" "${UPLOAD_USER}@${UPLOAD_HOST}:${REMOTE_TARGET}/"

echo "✅ Upload completed successfully."
