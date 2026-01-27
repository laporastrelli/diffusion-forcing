# Post-Training Methods: Hybrid History vs Self-Forcing

This document describes the two post-training methods now available in this codebase to address exposure bias in diffusion forcing models.

## Overview

Both methods aim to reduce the train-test distribution gap by training the model on its own predictions rather than ground truth. However, they differ significantly in complexity and implementation.

---

## Method 1: Hybrid History Training (Simplified)

**Implementation**: `_training_step_hybrid_history()` in `df_base.py`

### Algorithm

1. Generate a prefix sequence **once per batch** using standard diffusion forcing sampling
2. Concatenate generated prefix with ground-truth continuation
3. Train with standard MSE loss only on the continuation frames
4. Loss is masked to zero for the generated prefix

### Configuration

```yaml
algorithm.hybrid_history:
  enabled: true
  seed_frames: 10                    # GT frames to start generation
  generated_prefix_frames: 30        # Total prefix length (includes seed)
  conditioning_noise_level: 0        # Noise added to generated context (0=clean)
```

### Usage

```bash
bash scripts/posttrain_hybrid_history_single_pendulum_full.sh <checkpoint_path_or_wandb_run_id>
```

Or via Python:
```bash
python main.py \
  load=<checkpoint> \
  dataset=video_single_pendulum_full \
  experiment.tasks=["hybrid_history_posttrain"] \
  algorithm.hybrid_history.enabled=true \
  algorithm.hybrid_history.seed_frames=10 \
  algorithm.hybrid_history.generated_prefix_frames=30
```

### Characteristics

**Pros:**
- ✅ Simple and stable
- ✅ Computationally efficient
- ✅ Uses standard MSE loss (well-understood)
- ✅ No complex autoregressive loops
- ✅ Still addresses exposure bias

**Cons:**
- ❌ Not true autoregressive rollout
- ❌ Prefix generated only once per batch
- ❌ Less faithful to Self-Forcing paper
- ❌ May not fully simulate test-time generation

**Best for:** Quick experiments, stable training, limited compute

---

## Method 2: Self-Forcing (Autoregressive Rollout)

**Implementation**: `_training_step_self_forcing()` in `df_base.py`

### Algorithm

1. Start with GT seed frames
2. **Autoregressively generate** blocks of frames:
   - Initialize random noise for current block
   - Perform multi-step denoising (e.g., 4 steps)
   - Randomly select one timestep to backpropagate through
   - Use generated block as context for next block
3. Each block sees all previously generated blocks as context
4. Train with MSE loss on each generated block

### Configuration

```yaml
algorithm.self_forcing:
  enabled: true
  seed_frames: 10                    # GT frames to start autoregressive generation
  block_size: 4                      # Frames generated per block
  num_denoising_steps: 4             # Denoising steps per block
  random_exit: true                  # Randomly select timestep for gradients
  max_rollout_tokens: -1             # Max tokens to generate (-1 = all)
  conditioning_noise_level: 0        # Noise on generated context
```

### Usage

```bash
bash scripts/posttrain_self_forcing_single_pendulum_full.sh <checkpoint_path_or_wandb_run_id>
```

Or via Python:
```bash
python main.py \
  load=<checkpoint> \
  dataset=video_single_pendulum_full \
  experiment.tasks=["self_forcing_posttrain"] \
  algorithm.self_forcing.enabled=true \
  algorithm.self_forcing.seed_frames=10 \
  algorithm.self_forcing.block_size=4 \
  algorithm.self_forcing.num_denoising_steps=4
```

### Characteristics

**Pros:**
- ✅ True autoregressive rollout during training
- ✅ More faithful to Self-Forcing paper methodology
- ✅ Each block conditions on all previous blocks
- ✅ Random timestep exits (efficiency + exploration)
- ✅ Better simulates test-time generation

**Cons:**
- ❌ More computationally expensive
- ❌ More complex implementation
- ❌ Requires careful hyperparameter tuning
- ❌ May be less stable (more moving parts)

**Best for:** Maximum performance, research comparisons, faithful paper replication

---

## Key Differences Summary

| Aspect | Hybrid History | Self-Forcing |
|--------|----------------|--------------|
| **Generation** | Once per batch | Block-by-block autoregressive |
| **Context** | Static prefix | Accumulating across blocks |
| **Denoising** | Standard DF sampling | Multi-step per block |
| **Gradients** | All continuation frames | Random timestep per block |
| **Complexity** | Low | High |
| **Compute** | Fast | Slower |
| **Faithfulness to paper** | Low | High |

---

## Hyperparameter Tuning Guide

### For Hybrid History:

1. **seed_frames** (10-20): More GT context = more stable
2. **generated_prefix_frames** (20-40): Longer prefix = more exposure correction
3. **conditioning_noise_level** (0-50): 0 for test-time match, >0 for robustness

### For Self-Forcing:

1. **seed_frames** (10-20): Same as above
2. **block_size** (2-8): Smaller = more granular, larger = faster
3. **num_denoising_steps** (3-5): Balance quality vs compute
4. **random_exit** (true/false): true for efficiency, false for consistency
5. **max_rollout_tokens**: Limit rollout length to save compute

### Learning Rate:

Both methods typically use **lower LR** than base training:
- Base training: 8e-5
- Post-training: 8e-6 to 1e-5

### Training Duration:

Both methods typically need **short post-training**:
- 1 epoch often sufficient
- Monitor for overfitting

---

## Experimental Protocol

To compare both methods:

1. **Train baseline model** with standard diffusion forcing
2. **Post-train with Hybrid History**:
   ```bash
   bash scripts/posttrain_hybrid_history_single_pendulum_full.sh <checkpoint>
   ```
3. **Post-train with Self-Forcing** (from same baseline):
   ```bash
   bash scripts/posttrain_self_forcing_single_pendulum_full.sh <checkpoint>
   ```
4. **Evaluate all three** on long-horizon rollout tasks

### Metrics to Compare:

- MSE on short-horizon predictions
- MSE on long-horizon rollouts
- Stability (error accumulation over time)
- Training time
- Inference time (should be same for all)

---

## Technical Implementation Details

### Hybrid History Flow:

```
Batch → Generate prefix (no grad) → Concat with GT → Denoise (with grad) → Loss on continuation
```

### Self-Forcing Flow:

```
Batch → Seed frames (GT)
  → Block 1: Denoise (random step with grad) → Append to context
  → Block 2: Denoise using Block 1 context (random step with grad) → Append
  → Block 3: Denoise using Block 1+2 context (random step with grad) → Append
  → ...
  → Loss averaged across blocks
```

### Random Timestep Exit:

In Self-Forcing, each block undergoes multi-step denoising:
- Step 0: t=1000 → t=750 (no grad)
- Step 1: t=750 → t=500 (maybe grad)
- Step 2: t=500 → t=250 (maybe grad)  ← randomly selected
- Step 3: t=250 → t=0 (no grad)

Only one step per block computes gradients (efficiency trick from paper).

---

## Troubleshooting

### Hybrid History issues:

- **Loss not decreasing**: Increase generated_prefix_frames
- **Unstable training**: Reduce LR or add conditioning noise
- **No improvement over baseline**: Prefix too short or LR too low

### Self-Forcing issues:

- **OOM errors**: Reduce block_size or batch_size
- **Very slow**: Reduce num_denoising_steps or max_rollout_tokens
- **Unstable gradients**: Disable random_exit or reduce LR
- **Loss spikes**: Add conditioning noise or reduce max_rollout_tokens

---

## References

- **Diffusion Forcing paper**: https://arxiv.org/abs/2407.01392
- **Self-Forcing paper**: https://arxiv.org/abs/2506.08009
- **Original Self-Forcing repo**: https://github.com/guandeh17/Self-Forcing

---

## Code Locations

- **Algorithm implementations**: `algorithms/diffusion_forcing/df_base.py`
- **Configuration**: `configurations/algorithm/df_base.yaml`
- **Experiment tasks**: `experiments/exp_base.py`
- **Scripts**: `scripts/posttrain_*.sh`
