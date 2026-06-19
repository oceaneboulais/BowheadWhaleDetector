#!/bin/bash
set -e

# Start TensorBoard on an internal port (nginx proxies it on :7860).
tensorboard \
    --logdir  /app/logs \
    --host    127.0.0.1 \
    --port    7861 \
    --load_fast=false \
    &

# Give TensorBoard time to scan all event files before nginx accepts traffic.
sleep 20

# Start nginx in the foreground (keeps the container alive).
nginx -g "daemon off;"
