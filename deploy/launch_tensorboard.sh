#!/usr/bin/env bash
# Launch the training TensorBoard locally, with an optional instant public link.
#
#   ./deploy/launch_tensorboard.sh            # local only -> http://localhost:6006
#   ./deploy/launch_tensorboard.sh --public   # also opens a cloudflared quick tunnel
#                                              # -> https://<random>.trycloudflare.com
#
# The public link is EPHEMERAL (lives only while this script runs). For a durable
# portfolio URL, deploy the Hugging Face Space — see deploy/tensorboard_space/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="${REPO_ROOT}/runs"
PORT="${PORT:-6006}"
PY="${PYTHON:-/usr/local/bin/python3.8}"

if [ ! -d "$LOGDIR" ] || [ -z "$(ls -A "$LOGDIR" 2>/dev/null)" ]; then
  echo "No TensorBoard logs in ${LOGDIR}. Train first, e.g.:"
  echo "  PYTHONPATH=. ${PY} -m bowhead.train.train_cnn --data data/spectrograms.npz --tag scratch"
  exit 1
fi

echo "Starting TensorBoard on http://localhost:${PORT}  (logdir=${LOGDIR})"
"$PY" -m tensorboard.main --logdir "$LOGDIR" --host 0.0.0.0 --port "$PORT" &
TB_PID=$!
trap 'kill ${TB_PID} 2>/dev/null || true' EXIT

if [ "${1:-}" = "--public" ]; then
  command -v cloudflared >/dev/null || { echo "cloudflared not found on PATH"; exit 1; }
  echo "Opening public cloudflared tunnel (Ctrl-C to stop)…"
  cloudflared tunnel --url "http://localhost:${PORT}"
else
  echo "Local only. Re-run with --public for a temporary shareable link. Ctrl-C to stop."
  wait ${TB_PID}
fi
