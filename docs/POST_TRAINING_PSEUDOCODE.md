# Post-Training Methods Pseudo-Code

This document provides high-level pseudo-code for the two post-training methods implemented in the Diffusion-Forcing framework: **Hybrid History Training** and **Self-Forcing Training**.

## Configuration Summary

### Hybrid History Training Configuration
- **Script**: `scripts/posttrain_hybrid_history_single_pendulum_full.sh`
- **Key parameters**:
  - `seed_frames=10`: Number of ground-truth frames used as initial context
  - `generated_prefix_frames=30`: Total prefix length (includes seed + generated frames)
  - `conditioning_noise_level=0`: Noise level applied to generated prefix during training
  - `lr=8e-6`: Learning rate
  - `batch_size=3`: Batch size
  - `max_epochs=1`: Number of training epochs

### Self-Forcing Training Configuration
- **Script**: `scripts/posttrain_self_forcing_single_pendulum_full.sh`
- **Key parameters**:
  - `seed_frames=10`: Number of ground-truth frames used as initial context
  - `block_size=4`: Size of each autoregressive block
  - `num_denoising_steps=20`: Number of denoising steps per block
  - `random_exit=true`: Whether to randomly select which denoising step to backprop through
  - `max_rollout_tokens=-1`: Maximum rollout length (-1 means full sequence)
  - `conditioning_noise_level=0`: Noise level applied to generated context
  - `lr=8e-6`: Learning rate
  - `batch_size=3`: Batch size
  - `max_epochs=1`: Number of training epochs

---

## 1. Hybrid History Training (Simplified Exposure Correction)

**Purpose**: Train the model to continue generation after a generated (potentially imperfect) prefix, rather than only after perfect ground-truth context.

### High-Level Algorithm

```
FUNCTION hybrid_history_training_step(batch):
    """
    Hybrid History Training: Generate prefix once, concatenate with GT, train on continuation.
    This is a simplified exposure correction method that doesn't require full autoregressive rollout.
    """
    
    # === 1. PREPROCESSING ===
    xs, conditions, masks, idxs = preprocess_batch(batch)
    n_tokens, batch_size = xs.shape[0], xs.shape[1]
    
    # === 2. CONFIGURATION ===
    seed_frames = 10  # Ground-truth frames at start
    seed_tokens = seed_frames / frame_stack
    
    generated_prefix_frames = 30  # Total prefix length (seed + generated)
    prefix_tokens = generated_prefix_frames / frame_stack
    
    conditioning_noise_level = 0  # Noise added to prefix during training
    
    # === 3. GENERATE PREFIX (NO GRADIENT) ===
    # Sample a generated prefix using the model's autoregressive sampling
    # This keeps the first seed_tokens as ground-truth, generates the rest
    xs_prefix = sample_prefix_tokens(
        xs,
        conditions,
        n_context_tokens=seed_tokens,    # Use GT for first 10 frames
        n_total_tokens=prefix_tokens     # Generate up to 30 frames total
    )
    # Note: sample_prefix_tokens() uses standard diffusion sampling with no gradients
    
    # === 4. BUILD HYBRID SEQUENCE ===
    # Concatenate: [generated prefix (detached)] + [GT remainder]
    xs_hybrid = concatenate([
        xs_prefix.detach(),              # Frames 0-29: Generated (no gradient)
        xs[prefix_tokens:]               # Frames 30+: Ground-truth
    ])
    
    # === 5. GENERATE NOISE LEVELS ===
    # Standard random noise levels for all tokens
    noise_levels = generate_noise_levels(xs_hybrid)
    
    # Override: Set generated prefix to fixed conditioning noise level
    noise_levels[:prefix_tokens] = conditioning_noise_level  # Usually 0 = clean
    
    # === 6. FORWARD PASS (WITH GRADIENT) ===
    # Denoise the hybrid sequence (model sees generated context + noisy GT continuation)
    xs_pred, loss = diffusion_model(xs_hybrid, conditions, noise_levels)
    
    # === 7. REWEIGHT LOSS ===
    # Zero out loss for generated prefix, only learn on continuation
    weight = ones(n_tokens * frame_stack, batch_size)
    weight[:prefix_tokens * frame_stack] = 0.0
    loss = reweight_loss(loss, weight)
    
    # === 8. LOGGING & RETURN ===
    log("training/loss", loss)
    log("training/hybrid_history_seed_tokens", seed_tokens)
    log("training/hybrid_history_prefix_tokens", prefix_tokens)
    
    return {
        "loss": loss,
        "xs_pred": unstack_and_unnormalize(xs_pred),
        "xs": unstack_and_unnormalize(xs),
        "idxs": idxs
    }


FUNCTION sample_prefix_tokens(xs, conditions, n_context_tokens, n_total_tokens):
    """
    Sample tokens up to n_total_tokens, using first n_context_tokens as ground-truth.
    Uses standard Diffusion-Forcing sampling (no gradient tracking).
    """
    
    n_frames, batch_size = xs.shape[:2]
    n_total_tokens = min(n_total_tokens, n_frames)
    n_context_tokens = min(max(n_context_tokens, 0), n_total_tokens)
    
    # Start with ground-truth context
    xs_pred = xs[:n_context_tokens].clone()
    curr_frame = n_context_tokens
    
    # Autoregressive generation with no gradients
    while curr_frame < n_total_tokens:
        # Determine horizon (chunk size for this iteration)
        if chunk_size > 0:
            horizon = min(n_total_tokens - curr_frame, chunk_size)
        else:
            horizon = n_total_tokens - curr_frame
        
        # Generate diffusion scheduling matrix
        scheduling_matrix = generate_scheduling_matrix(horizon)
        
        # Initialize random noise for new chunk
        chunk = random_noise(horizon, batch_size, device)
        chunk = clamp(chunk, -clip_noise, clip_noise)
        xs_pred = concatenate([xs_pred, chunk])
        
        # Sliding window: only process last n_tokens frames
        start_frame = max(0, curr_frame + horizon - n_tokens)
        
        # Iterative denoising using scheduling matrix
        for m in range(scheduling_matrix.shape[0] - 1):
            # Build noise level arrays
            from_noise_levels = concatenate([
                zeros(curr_frame),           # Clean context
                scheduling_matrix[m]         # Noisy new frames
            ]).repeat_for_batch(batch_size)
            
            to_noise_levels = concatenate([
                zeros(curr_frame),           # Clean context
                scheduling_matrix[m + 1]     # Next noise level
            ]).repeat_for_batch(batch_size)
            
            # Denoise step (sliding window)
            xs_pred[start_frame:] = diffusion_model.sample_step(
                xs_pred[start_frame:],
                conditions[start_frame : curr_frame + horizon],
                from_noise_levels[start_frame:],
                to_noise_levels[start_frame:]
            )
        
        curr_frame += horizon
    
    return xs_pred[:n_total_tokens]
```

### Key Points for Hybrid History:
1. **One-shot prefix generation**: Generate the prefix once at the start (no gradients)
2. **Hybrid sequence**: Concatenate generated prefix (detached) + GT continuation
3. **Loss masking**: Only compute loss on frames after the generated prefix
4. **Fixed noise level**: Generated prefix is typically set to noise level 0 (clean conditioning)
5. **Efficient**: Only one forward pass with gradients per batch

---

## 2. Self-Forcing Training (Full Autoregressive Rollout)

**Purpose**: Train through full autoregressive rollout with block-by-block generation, backpropagating through randomly selected denoising steps to improve long-term generation quality.

### High-Level Algorithm

```
FUNCTION self_forcing_training_step(batch):
    """
    True Self-Forcing: Autoregressive block-by-block generation with random timestep exits.
    Backpropagates through the denoising process at randomly selected steps.
    """
    
    # === 1. PREPROCESSING ===
    xs, conditions, masks, idxs = preprocess_batch(batch)
    n_tokens, batch_size = xs.shape[0], xs.shape[1]
    
    # === 2. CONFIGURATION ===
    seed_frames = 10  # Ground-truth frames at start
    seed_tokens = seed_frames / frame_stack
    
    block_size = 4  # Generate 4 frames at a time
    block_tokens = block_size / frame_stack
    
    num_denoising_steps = 20  # Number of diffusion steps per block
    random_exit = true  # Randomly select which step to backprop through
    conditioning_noise_level = 0  # Noise on generated context
    
    max_rollout_tokens = n_tokens  # Generate full sequence
    
    # === 3. INITIALIZE ROLLOUT ===
    xs_rollout = xs[:seed_tokens].clone()  # Start with GT seed
    current_token = seed_tokens
    
    total_loss = 0.0
    num_loss_blocks = 0
    
    # === 4. AUTOREGRESSIVE BLOCK-BY-BLOCK GENERATION ===
    while current_token < max_rollout_tokens:
        
        # === 4.1 DETERMINE BLOCK SIZE ===
        tokens_remaining = max_rollout_tokens - current_token
        current_block_size = min(block_tokens, tokens_remaining)
        
        # === 4.2 SELECT RANDOM EXIT STEP ===
        if random_exit:
            exit_step = random_int(0, num_denoising_steps)
        else:
            exit_step = num_denoising_steps - 1
        
        # === 4.3 INITIALIZE NOISY BLOCK ===
        block_noise = random_noise(current_block_size, batch_size, device)
        block_noise = clamp(block_noise, -clip_noise, clip_noise)
        denoised_block = block_noise
        
        # === 4.4 GENERATE SCHEDULING MATRIX ===
        scheduling_matrix = generate_scheduling_matrix(current_block_size)
        
        # Subsample to exactly num_denoising_steps
        row_idxs = linspace(0, scheduling_matrix.shape[0] - 1, 
                           num_denoising_steps + 1)
        
        # === 4.5 GROUND-TRUTH FOR THIS BLOCK ===
        gt_block = xs[current_token : current_token + current_block_size]
        
        # === 4.6 SLIDING WINDOW START POSITION ===
        start_token = max(0, current_token + current_block_size - n_tokens)
        
        # === 4.7 MULTI-STEP DENOISING ===
        for step_idx in range(num_denoising_steps):
            
            # Get current and next noise schedules
            curr_sched = scheduling_matrix[row_idxs[step_idx]]
            next_sched = scheduling_matrix[row_idxs[step_idx + 1]]
            
            # === 4.7.1 CONCATENATE CONTEXT + CURRENT BLOCK ===
            xs_input = concatenate([xs_rollout, denoised_block])
            
            # === 4.7.2 BUILD NOISE LEVEL ARRAYS ===
            from_noise_levels = concatenate([
                zeros(current_token),        # Clean/noisy context
                curr_sched                   # Current block noise
            ]).repeat_for_batch(batch_size)
            
            to_noise_levels = concatenate([
                zeros(current_token),        # Clean/noisy context
                next_sched                   # Next block noise
            ]).repeat_for_batch(batch_size)
            
            # === 4.7.3 OPTIONAL: ADD NOISE TO GENERATED CONTEXT ===
            # (Not to GT seed, only to previously generated tokens)
            if current_token > seed_tokens and conditioning_noise_level > 0:
                from_noise_levels[seed_tokens:current_token] = conditioning_noise_level
                to_noise_levels[seed_tokens:current_token] = conditioning_noise_level
            
            # === 4.7.4 EXTRACT CONDITIONS FOR SLIDING WINDOW ===
            if conditions is not None:
                cond_window = conditions[start_token : current_token + current_block_size]
            else:
                cond_window = None
            
            # === 4.7.5 DECIDE WHETHER TO COMPUTE GRADIENTS ===
            compute_gradients = (step_idx == exit_step) and (current_token >= seed_tokens)
            
            if compute_gradients:
                # === GRADIENT STEP ===
                # Denoise with gradient tracking (backprop through this step)
                xs_input[start_token:] = diffusion_model.sample_step(
                    xs_input[start_token:],
                    cond_window,
                    from_noise_levels[start_token:],
                    to_noise_levels[start_token:]
                )
                denoised_block = xs_input[-current_block_size:]
                
                # Compute loss against ground-truth
                block_loss = mse_loss(denoised_block, gt_block)
                total_loss += block_loss
                num_loss_blocks += 1
                
                # Exit denoising loop after computing gradient
                break
            
            else:
                # === NO-GRADIENT STEP ===
                # Denoise without gradient tracking
                with no_grad():
                    xs_input[start_token:] = diffusion_model.sample_step(
                        xs_input[start_token:],
                        cond_window,
                        from_noise_levels[start_token:],
                        to_noise_levels[start_token:]
                    )
                    denoised_block = xs_input[-current_block_size:].detach()
        
        # === 4.8 APPEND GENERATED BLOCK TO ROLLOUT ===
        # Detach so next block uses it as fixed context (no gradient flow back)
        xs_rollout = concatenate([xs_rollout, denoised_block.detach()])
        current_token += current_block_size
    
    # === 5. AVERAGE LOSS ACROSS BLOCKS ===
    if num_loss_blocks > 0:
        loss = total_loss / num_loss_blocks
    else:
        loss = total_loss
    
    # === 6. LOGGING & RETURN ===
    log("training/loss", loss)
    log("training/self_forcing_blocks", num_loss_blocks)
    log("training/self_forcing_rollout_tokens", current_token)
    
    return {
        "loss": loss,
        "xs_pred": unstack_and_unnormalize(xs_rollout[:n_tokens]),
        "xs": unstack_and_unnormalize(xs),
        "idxs": idxs
    }
```

### Key Points for Self-Forcing:
1. **Block-by-block autoregressive**: Generate sequence incrementally in blocks of `block_size` frames
2. **Multi-step denoising**: Each block undergoes `num_denoising_steps` diffusion steps
3. **Random exit**: Randomly select which denoising step to backprop through (improves robustness)
4. **Gradient flow**: Only compute gradients at the selected exit step, all other steps are no_grad()
5. **Detached context**: Each generated block is detached before being used as context for the next block
6. **Sliding window**: Uses standard Diffusion-Forcing sliding window attention (last `n_tokens` frames)
7. **Loss per block**: Compute MSE loss between denoised block and ground-truth for each block
8. **No loss on seed**: Skip gradient computation for the initial seed frames (always GT)

---

## Comparison

| Aspect | Hybrid History | Self-Forcing |
|--------|----------------|--------------|
| **Complexity** | Simpler, one-shot prefix generation | More complex, full autoregressive rollout |
| **Generation** | Generate prefix once, detach, then train on continuation | Generate each block autoregressively during training |
| **Gradients** | Only through continuation (after prefix) | Through randomly selected denoising steps in each block |
| **Computational Cost** | Lower (one forward pass) | Higher (multiple denoising steps per block) |
| **Exposure Correction** | Simplified: single generated prefix | Full: accumulates errors throughout rollout |
| **Loss Computation** | Loss only on frames after generated prefix | Loss on each generated block |
| **Training Stability** | More stable (fixed prefix per batch) | Less stable (errors accumulate) but more realistic |
| **Memory Usage** | Lower | Higher (stores gradients through denoising) |

---

## Implementation Notes

### Hybrid History
- The generated prefix is created using the **same sampling procedure** as validation/inference
- The prefix is **detached** from the computation graph (no gradients flow back)
- Loss is **masked** to zero for all prefix frames, only learning on continuation
- The model sees: `[clean generated frames] + [noisy GT frames]`
- Prefix noise level is typically 0 (clean), but can be increased for robustness

### Self-Forcing
- Uses the **diffusion scheduling matrix** to determine noise levels at each step
- The `random_exit` strategy randomly selects which denoising step to backprop through
- This prevents overfitting to a specific denoising trajectory
- Each block's denoising is **interrupted** at the exit step to compute gradients
- Previous denoising steps are done with `no_grad()` to save memory
- The generated blocks are **immediately detached** after gradient computation
- Uses the same **sliding window mechanism** as standard Diffusion-Forcing

### Common Elements
- Both use the same underlying diffusion model and sampling logic
- Both respect the `frame_stack` parameter (multiple frames per token)
- Both use sliding window attention (`n_tokens` context limit)
- Both can optionally add noise to generated context via `conditioning_noise_level`
- Both are post-training methods: they fine-tune a pre-trained Diffusion-Forcing model

---

## Configuration Recommendations

### Hybrid History
- `seed_frames`: 10-20 (enough to establish dynamics)
- `generated_prefix_frames`: 20-40 (moderate exposure to generated context)
- `conditioning_noise_level`: 0 (clean) or small values for robustness
- `lr`: 1e-6 to 1e-5 (lower than pre-training)
- `max_epochs`: 1-3 (quick fine-tuning)

### Self-Forcing
- `seed_frames`: 10-20 (same as hybrid history)
- `block_size`: 2-8 frames (smaller = more stable, larger = faster)
- `num_denoising_steps`: 10-20 (trade-off: quality vs. speed)
- `random_exit`: true (recommended for robustness)
- `conditioning_noise_level`: 0 or small values
- `lr`: 1e-6 to 1e-5 (lower than pre-training)
- `max_epochs`: 1-3 (computationally expensive)

---

## References

Implementation files:
- Main entry point: `main.py`
- Task execution: `experiments/exp_base.py` (lines 235-410)
- Algorithm implementation: `algorithms/diffusion_forcing/df_base.py` (lines 111-410)
- Configuration: `scripts/posttrain_hybrid_history_single_pendulum_full.sh`
- Configuration: `scripts/posttrain_self_forcing_single_pendulum_full.sh`
