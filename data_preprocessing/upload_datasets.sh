#!/usr/bin/env bash
set -euo pipefail
trap 'echo "✖ Upload interrupted."; exit 1' INT

HERE="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${HERE}/.." && pwd)"
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a; # shellcheck source=/dev/null
  source "${REPO_ROOT}/.env"
  set +a
fi

: "${REMOTE_USER:?Set REMOTE_USER in .env}"
: "${REMOTE_HOST:?Set REMOTE_HOST in .env}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-${REPO_ROOT}/data}"

LOCAL_ROOT="${1:-}"
if [[ -z "$LOCAL_ROOT" ]]; then
  echo "Usage: ./data_preprocessing/upload_datasets.sh <local_folder_path>" >&2
  exit 1
fi
if [[ ! -d "$LOCAL_ROOT" ]]; then
  echo "✖ '$LOCAL_ROOT' is not a directory or does not exist." >&2
  exit 1
fi

LOCAL_ROOT_ABS="$(cd "$LOCAL_ROOT" && pwd)"
FOLDER_NAME="$(basename "$LOCAL_ROOT_ABS")"
REMOTE_TARGET_PATH="${REMOTE_DATA_ROOT}/${FOLDER_NAME}"

echo "📦 Will upload:"
echo "   ${LOCAL_ROOT_ABS}"
echo "🛰️  → ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_TARGET_PATH}"
ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '${REMOTE_TARGET_PATH}'"
rsync -av --partial --progress -e ssh "${LOCAL_ROOT_ABS}/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_TARGET_PATH}/"
echo "✅ Upload completed."
