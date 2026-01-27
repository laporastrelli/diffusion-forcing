#!/usr/bin/env bash
set -euo pipefail

# Post-train a Diffusion-Forcing single pendulum model with Self-Forcing (autoregressive rollout).
# This implements the full autoregressive block-by-block generation during training.
#
# Usage:
#   bash scripts/posttrain_self_forcing_single_pendulum_full.sh /path/to/model.ckpt
# or (wandb run id):
#   bash scripts/posttrain_self_forcing_single_pendulum_full.sh pxzudju2

name="df_single_pendulum_full_self_forcing_posttrain"
load=qtmk1xo7
dataset="video_single_pendulum_full"

CUDA_VISIBLE_DEVICES=0 python main.py \
  +name=${name} \
  load=${load} \
  dataset=${dataset} \
  experiment.tasks=["self_forcing_posttrain","validation"] \
  experiment.self_forcing_posttrain.lr=8e-6 \
  experiment.self_forcing_posttrain.max_epochs=1 \
  experiment.self_forcing_posttrain.batch_size=3 \
  algorithm.self_forcing.enabled=true \
  algorithm.self_forcing.seed_frames=10 \
  algorithm.self_forcing.block_size=1 \
  algorithm.self_forcing.num_denoising_steps=20 \
  algorithm.self_forcing.random_exit=true \
  algorithm.self_forcing.max_rollout_tokens=-1 \
  algorithm.self_forcing.conditioning_noise_level=0 \
  algorithm.metrics=["mse"]
