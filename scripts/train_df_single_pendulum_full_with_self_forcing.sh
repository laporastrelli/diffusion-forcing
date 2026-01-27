#!/usr/bin/env bash
set -euo pipefail

# Train Diffusion-Forcing on single pendulum full, then run a Self-Forcing post-training stage.
# This is the fastest path that stays inside the diffusion-forcing codebase.

name="df_single_pendulum_full_self_forcing"
dataset="video_single_pendulum_full"

# Match your original training script defaults
batch_size=3
epochs=100
checkpointing_frequency=20
val_every_n_epoch=102
val_batch_size=20

# Self-Forcing stage defaults (tune as needed)
sf_lr=8e-6
sf_epochs=1
sf_seed_frames=10
sf_prefix_frames=30

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3} python main.py \
  +name=${name} \
  dataset=${dataset} \
  experiment.training.batch_size=${batch_size} \
  experiment.training.max_epochs=${epochs} \
  experiment.training.checkpointing.every_n_epochs=${checkpointing_frequency} \
  experiment.validation.val_every_n_epoch=${val_every_n_epoch} \
  experiment.validation.batch_size=${val_batch_size} \
  experiment.tasks=["training","self_forcing_posttrain","validation"] \
  experiment.self_forcing_posttrain.lr=${sf_lr} \
  experiment.self_forcing_posttrain.max_epochs=${sf_epochs} \
  experiment.self_forcing_posttrain.batch_size=${batch_size} \
  algorithm.self_forcing.enabled=true \
  algorithm.self_forcing.seed_frames=${sf_seed_frames} \
  algorithm.self_forcing.generated_prefix_frames=${sf_prefix_frames} \
  algorithm.self_forcing.conditioning_noise_level=0 \
  algorithm.metrics=["mse"]
