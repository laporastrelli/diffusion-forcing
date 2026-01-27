#!/usr/bin/env bash
set -euo pipefail

# Post-train a Diffusion-Forcing single pendulum model with Hybrid History Training.
# This is the simplified exposure correction method (not full autoregressive rollout).
#
# Usage:
#   bash scripts/posttrain_hybrid_history_single_pendulum_full.sh /path/to/model.ckpt
# or (wandb run id):
#   bash scripts/posttrain_hybrid_history_single_pendulum_full.sh pxzudju2

load="${1:-}"
if [[ -z "${load}" ]]; then
  echo "Usage: $0 <checkpoint_path_or_wandb_run_id>" >&2
  exit 1
fi

name="df_single_pendulum_full_hybrid_history_posttrain"
dataset="video_single_pendulum_full"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python main.py \
  +name=${name} \
  load=${load} \
  dataset=${dataset} \
  experiment.tasks=["hybrid_history_posttrain"] \
  experiment.hybrid_history_posttrain.lr=8e-6 \
  experiment.hybrid_history_posttrain.max_epochs=1 \
  experiment.hybrid_history_posttrain.batch_size=3 \
  algorithm.hybrid_history.enabled=true \
  algorithm.hybrid_history.seed_frames=10 \
  algorithm.hybrid_history.generated_prefix_frames=30 \
  algorithm.hybrid_history.conditioning_noise_level=0 \
  algorithm.metrics=["mse"]
