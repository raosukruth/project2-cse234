#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-rftrain}"
CONDA_ENV="${2:-cse234}"
NOTEBOOK_PATH="${3:-rf_experiments.ipynb}"
LOG_PATH="${4:-rf_experiments.log}"
OUTPUT_NOTEBOOK="${5:-rf_experiments_ran.ipynb}"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v tmux >/dev/null 2>&1; then
  echo "Error: tmux is not installed. Install tmux first."
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda is not available in PATH."
  exit 1
fi

if [[ ! -f "${WORKDIR}/${NOTEBOOK_PATH}" ]]; then
  echo "Error: notebook not found at ${WORKDIR}/${NOTEBOOK_PATH}"
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "Using existing tmux session: ${SESSION_NAME}"
else
  tmux new-session -d -s "${SESSION_NAME}"
  echo "Created tmux session: ${SESSION_NAME}"
fi

CONDA_BASE="$(conda info --base)"
RUN_CMD="cd '${WORKDIR}' && source '${CONDA_BASE}/etc/profile.d/conda.sh' && conda activate '${CONDA_ENV}' && nohup jupyter nbconvert --to notebook --execute '${NOTEBOOK_PATH}' --output '${OUTPUT_NOTEBOOK}' --ExecutePreprocessor.timeout=-1 > '${LOG_PATH}' 2>&1 & echo 'Notebook execution started in background. PID:' \$!"

tmux send-keys -t "${SESSION_NAME}" "${RUN_CMD}" C-m

echo
echo "Started notebook in background."
echo "Session: ${SESSION_NAME}"
echo "Log: ${WORKDIR}/${LOG_PATH}"
echo "Output notebook: ${WORKDIR}/${OUTPUT_NOTEBOOK}"
echo
echo "Useful commands:"
echo "  tmux attach -t ${SESSION_NAME}"
echo "  tail -f ${WORKDIR}/${LOG_PATH}"
