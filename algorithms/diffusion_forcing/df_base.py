"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

from omegaconf import DictConfig
import numpy as np
from random import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from einops import rearrange

import time

from lightning.pytorch.utilities.types import STEP_OUTPUT

from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from utils.logging_utils import get_validation_metrics_for_states
from .models.diffusion_transition import DiffusionTransitionModel

import sys

class DiffusionForcingBase(BasePytorchAlgo):
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.x_shape = cfg.x_shape
        self.z_shape = cfg.z_shape
        self.frame_stack = cfg.frame_stack
        self.cfg.diffusion.cum_snr_decay = self.cfg.diffusion.cum_snr_decay**self.frame_stack
        self.x_stacked_shape = list(cfg.x_shape)
        self.x_stacked_shape[0] *= cfg.frame_stack
        self.is_spatial = len(self.x_shape) == 3  # pixel
        self.gt_cond_prob = cfg.gt_cond_prob  # probability to condition one-step diffusion o_t+1 on ground truth o_t
        self.gt_first_frame = cfg.gt_first_frame
        self.context_frames = cfg.context_frames  # number of context frames at validation time
        self.chunk_size = cfg.chunk_size
        self.calc_crps_sum = cfg.calc_crps_sum
        self.external_cond_dim = cfg.external_cond_dim
        self.uncertainty_scale = cfg.uncertainty_scale
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.validation_step_outputs = []
        self.min_crps_sum = float("inf")
        self.learnable_init_z = cfg.learnable_init_z

        ##### new additions #####

        # new assertions for validation settings
        assert self.chunk_size >= 1, "Chunk size must be greater than 1 for validation tasks"
        assert self.context_frames + self.chunk_size <= cfg.n_frames, \
            "Context frames + chunk size must be <= total number of frames"

        # new option: perform imputation as validation (in addition to forecasting) 
        self.imputation_as_val = cfg.diffusion.get("imputation_as_val", False) # default: False
        self.symmetric_context = cfg.diffusion.get("symmetric_context", False) # default: False
            
        if self.imputation_as_val:
            if not self.symmetric_context:
                raise NotImplementedError("Asymmetric context not implemented for imputation yet")
            assert self.context_frames > 0, \
                "Context frames must be > 0 for imputation"
            assert self.imputation_as_val, \
                "symmetric_context can only be used with imputation_as_val"
            assert self.context_frames >= 2, \
                "context_frames must be even when using symmetric_context"
            assert self.context_frames % 2 == 0, \
                "context_frames must be even when using symmetric_context"
            assert self.context_frames + self.chunk_size == cfg.n_frames, \
                "Context frames + chunk size must be equal to total number of frames for imputation"
        
        super().__init__(cfg)


    def _is_bi_gru(self):
        if "bi-gru" in self.cfg.diffusion.transition_type.lower():
            return True
        return False

    def _build_model(self):
        self.transition_model = DiffusionTransitionModel(
            self.x_stacked_shape, self.z_shape, self.external_cond_dim, self.cfg.diffusion
        )
        self.register_data_mean_std(self.cfg.data_mean, self.cfg.data_std)
        if self.learnable_init_z:
            if not self._is_bi_gru():
                self.init_z = nn.Parameter(torch.randn(list(self.z_shape)), requires_grad=True)

    def configure_optimizers(self):
        transition_params = list(self.transition_model.parameters())
        if self.learnable_init_z:
            if not self._is_bi_gru():
                transition_params.append(self.init_z)
        optimizer_dynamics = torch.optim.AdamW(
            transition_params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, betas=self.cfg.optimizer_beta
        )

        return optimizer_dynamics

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # manually warm up lr without a scheduler
        if self.trainer.global_step < self.cfg.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.cfg.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.cfg.lr

    def _preprocess_batch(self, batch):
        xs = batch[0]
        batch_size, n_frames = xs.shape[:2]

        if n_frames % self.frame_stack != 0:
            raise ValueError("Number of frames must be divisible by frame stack size")
        if self.context_frames % self.frame_stack != 0:
            raise ValueError("Number of context frames must be divisible by frame stack size")

        nonterminals = batch[-1]
        nonterminals = nonterminals.bool().permute(1, 0)
        masks = torch.cumprod(nonterminals, dim=0).contiguous()
        n_frames = n_frames // self.frame_stack

        if self.external_cond_dim:
            conditions = batch[1]
            conditions = torch.cat([torch.zeros_like(conditions[:, :1]), conditions[:, 1:]], 1)
            conditions = rearrange(conditions, "b (t fs) d -> t b (fs d)", fs=self.frame_stack).contiguous()
        else:
            conditions = [None for _ in range(n_frames)]

        xs = self._normalize_x(xs)
        if self._is_bi_gru():
            xs = rearrange(xs, "t b c ... -> b t c ...")  # [B,T,C,...]
        else:   
            xs = rearrange(xs, "b (t fs) c ... -> t b (fs c) ...", fs=self.frame_stack).contiguous()

        if self.learnable_init_z:
            if self._is_bi_gru():
                init_z = None # bi-gru initial states are handled inside the model
            else:
                init_z = self.init_z[None].expand(batch_size, *self.z_shape)
        else:
            init_z = torch.zeros(batch_size, *self.z_shape)
            init_z = init_z.to(xs.device)

        return xs, conditions, masks, init_z

    def reweigh_loss(self, loss, weight=None):
        loss = rearrange(loss, "t b (fs c) ... -> t b fs c ...", fs=self.frame_stack)
        if weight is not None:
            expand_dim = len(loss.shape) - len(weight.shape) - 1
            weight = rearrange(weight, "(t fs) b ... -> t b fs ..." + " 1" * expand_dim, fs=self.frame_stack)
            loss = loss * weight

        return loss.mean()

    def training_step(self, batch, batch_idx):
        # training step for dynamics
        xs, conditions, masks, *_, init_z = self._preprocess_batch(batch)
        
        n_frames, _, _, *_ = xs.shape

        if self._is_bi_gru():
            # reshape time-first -> batch-first so the transition can see the whole sequence
            xs_seq = rearrange(xs, "t b c ... -> b t c ...")  # [B,T,(fs*C),...]
            if self.external_cond_dim:
                cond_seq = rearrange(conditions, "t b d -> b t d")
            else:
                cond_seq = None

            # single pass through the transition model in sequence mode
            # It returns x_next_seq, loss_seq, cum_snr_seq (per-frame)
            _, x_next_seq, loss_seq, _ = self.transition_model(
                None, xs_seq, cond_seq, deterministic_t=None, cum_snr=None
            )

            # back to time-first for weighting
            xs_pred = rearrange(x_next_seq, "b t c ... -> t b c ...")
            loss    = rearrange(loss_seq,    "b t c ... -> t b c ...")

            x_loss  = self.reweigh_loss(loss, masks)
            loss    = x_loss

            if batch_idx % 20 == 0:
                self.log_dict({"training/loss": loss, "training/x_loss": x_loss})

            xs_flat      = rearrange(xs,      "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
            xs_pred_flat = rearrange(xs_pred, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)

            output_dict = {
                "loss": loss,
                "xs_pred": self._unnormalize_x(xs_pred_flat),
                "xs": self._unnormalize_x(xs_flat),
            }
            return output_dict

        else:
            xs_pred = []
            loss = []
            z = init_z
            cum_snr = None
            for t in range(0, n_frames):
                deterministic_t = None
                if random() <= self.gt_cond_prob or (t == 0 and random() <= self.gt_first_frame):
                    deterministic_t = 0

                z_next, x_next_pred, l, cum_snr = self.transition_model(
                    z, xs[t], conditions[t], deterministic_t=deterministic_t, cum_snr=cum_snr
                )

                z = z_next
                xs_pred.append(x_next_pred)
                loss.append(l)

            xs_pred = torch.stack(xs_pred)
            loss = torch.stack(loss)
            x_loss = self.reweigh_loss(loss, masks)
            loss = x_loss

            if batch_idx % 20 == 0:
                self.log_dict(
                    {
                        "training/loss": loss,
                        "training/x_loss": x_loss,
                    }
                )

            xs = rearrange(xs, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
            xs_pred = rearrange(xs_pred, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)

            output_dict = {
                "loss": loss,
                "xs_pred": self._unnormalize_x(xs_pred),
                "xs": self._unnormalize_x(xs),
            }

            return output_dict

    @torch.no_grad()
    def validation_step_og(self, batch, batch_idx, namespace="validation"):
        if self.calc_crps_sum:
            # repeat batch for crps sum for time series prediction
            batch = [d[None].expand(self.calc_crps_sum, *([-1] * len(d.shape))).flatten(0, 1) for d in batch]

        xs, conditions, masks, *_, init_z = self._preprocess_batch(batch)

        n_frames, batch_size, *_ = xs.shape
        xs_pred = []
        xs_pred_all = []
        z = init_z

        # context
        for t in range(0, self.context_frames // self.frame_stack):
            z, x_next_pred, _, _ = self.transition_model(z, xs[t], conditions[t], deterministic_t=0)
            xs_pred.append(x_next_pred)

        # prediction (chunk-wise)
        while len(xs_pred) < n_frames:

            # set horizon
            if self.chunk_size > 0:
                horizon = min(n_frames - len(xs_pred), self.chunk_size)
            else:
                horizon = n_frames - len(xs_pred)

            # set chunk of predictions initialize to random noise with shape 
            # [ (B, C, H, W ), ..., (B, C, H, W) ] of length horizon
            chunk = [
                torch.randn((batch_size,) + tuple(self.x_stacked_shape), device=self.device) for _ in range(horizon)
            ]

            # build pyramid timesteps for uncertainty-aware sampling
            pyramid_height = self.sampling_timesteps + int(horizon * self.uncertainty_scale)
            pyramid = np.zeros((pyramid_height, horizon), dtype=int)
            
            # populate pyramid with timesteps adjusted for uncertainty scaling
            for m in range(pyramid_height):
                for t in range(horizon):
                    pyramid[m, t] = m - int(t * self.uncertainty_scale)
            pyramid = np.clip(pyramid, a_min=0, a_max=self.sampling_timesteps, dtype=int)

            # initialize previous timestep vector for redundancy check
            i_prev_vec = -100 * torch.zeros_like(
                torch.as_tensor(pyramid[m], device=self.device, dtype=torch.long)
            )  # [horizon]
            
            # iterate over pyramid levels
            for m in range(pyramid_height):
                if self.transition_model.return_all_timesteps:
                    xs_pred_all.append(chunk)

                z_chunk = z.detach()

                '''
                print('#######################################', file=sys.stderr)
                print('Pyramid level:', m, file=sys.stderr)
                print('Row Pyramid entries:', pyramid[m], file=sys.stderr)
                print('Previous timestep vector:', i_prev_vec, file=sys.stderr)
                print('#######################################', file=sys.stderr)
                '''

                # iterate over time steps in the chunk
                for t in range(horizon):
                    i = min(pyramid[m, t], self.sampling_timesteps - 1)

                    '''
                    # check for redundant timestep
                    if m>0 and i == i_prev_vec[t]:
                        # skip redundant (and inconsistent) denoising step
                        print("------------------------------------------------------------", file=sys.stderr)
                        print(f'Skipping redundant timestep at chunk time {t}, timestep {i}', file=sys.stderr)
                        print("------------------------------------------------------------", file=sys.stderr)
                        continue  
                    '''                  

                    # perform DDIM sampling step if not redundant  
                    chunk[t], z_chunk = self.transition_model.ddim_sample_step(
                        chunk[t], z_chunk, conditions[len(xs_pred) + t], i
                    )

                    # update previous timestep vector
                    i_prev_vec[t] = i

                    # theoretically, one shall feed new chunk[t] with last z_chunk into transition model again 
                    # to get the posterior z_chunk, and optionaly, with small noise level k>0 for stablization. 
                    # However, since z_chunk in the above line already contains info about updated chunk[t] in 
                    # our simplied math model, we deem it suffice to directly take this z_chunk estimated from 
                    # last z_chunk and noiser chunk[t]. This saves half of the compute from posterior steps. 
                    # The effect of the above simplification already contains stablization: we always stablize 
                    # (ddim_sample_step is never called with noise level k=0 above)

            z = z_chunk
            xs_pred += chunk

        xs_pred = torch.stack(xs_pred)
        loss = F.mse_loss(xs_pred, xs, reduction="none")
        loss = self.reweigh_loss(loss, masks)

        xs = rearrange(xs, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        xs_pred = rearrange(xs_pred, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)

        xs = self._unnormalize_x(xs)
        xs_pred = self._unnormalize_x(xs_pred)

        if not self.is_spatial:
            if self.transition_model.return_all_timesteps:
                xs_pred_all = [torch.stack(item) for item in xs_pred_all]
                limit = self.transition_model.sampling_timesteps
                for i in np.linspace(1, limit, 5, dtype=int):
                    xs_pred = xs_pred_all[i]
                    xs_pred = self._unnormalize_x(xs_pred)
                    metric_dict = get_validation_metrics_for_states(xs_pred, xs)
                    self.log_dict(
                        {f"{namespace}/{i}_sampling_steps_{k}": v for k, v in metric_dict.items()},
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                    )
            else:
                metric_dict = get_validation_metrics_for_states(xs_pred, xs)
                self.log_dict(
                    {f"{namespace}/{k}": v for k, v in metric_dict.items()},
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )

        self.validation_step_outputs.append((xs_pred.detach().cpu(), xs.detach().cpu()))

        return loss

    #############################################################################################
    # new validation step with chunk-wise Bi-GRU sampler (First version)
    @torch.no_grad()
    def validation_step_v1(self, batch, batch_idx, namespace="validation"):
        if self.calc_crps_sum:
            # repeat batch for crps sum for time series prediction
            batch = [d[None].expand(self.calc_crps_sum, *([-1] * len(d.shape))).flatten(0, 1) for d in batch]

        # get preprocessed data
        xs, conditions, masks, *_, init_z = self._preprocess_batch(batch)

        # initialize helpers
        n_frames, batch_size, *_ = xs.shape
        xs_pred = []
        xs_pred_all = []

        # ---------------- context ----------------
        ctx_steps = self.context_frames // self.frame_stack
        if not self._is_bi_gru():
            # original GRU context rollout (deterministic)
            z = init_z
            for t in range(0, ctx_steps):
                z, x_next_pred, _, _ = self.transition_model(z, xs[t], conditions[t], deterministic_t=0)
                xs_pred.append(x_next_pred)
        else:
            # BiGRU: treat context as given (teacher-forced), no running z needed
            for t in range(0, ctx_steps):
                xs_pred.append(xs[t])

        # ---------------- prediction ----------------
        while len(xs_pred) < n_frames:
            # horizon
            if self.chunk_size > 0:
                horizon = min(n_frames - len(xs_pred), self.chunk_size)
            else:
                horizon = n_frames - len(xs_pred)

            # initialize chunk as random noise: [B, horizon, (fs*C), H, W]
            chunk = torch.stack(
                [torch.randn((batch_size,) + tuple(self.x_stacked_shape), device=self.device) for _ in range(horizon)],
                dim=1
            )

            # pyramid timesteps for uncertainty-aware sampling
            pyramid_height = self.sampling_timesteps + int(horizon * self.uncertainty_scale)
            pyramid = np.zeros((pyramid_height, horizon), dtype=int)
            
            # populate pyramid with timesteps adjusted for uncertainty scaling
            for m in range(pyramid_height):
                for t in range(horizon):
                    pyramid[m, t] = m - int(t * self.uncertainty_scale)
            pyramid = np.clip(pyramid, a_min=0, a_max=self.sampling_timesteps, dtype=int)

            if not self._is_bi_gru():
                # ---------------- original GRU per-frame sampler ----------------
                z_chunk = z.detach()
                for m in range(pyramid_height):
                    if self.transition_model.return_all_timesteps:
                        xs_pred_all.append([chunk[:, t].clone() for t in range(horizon)])

                    for t in range(horizon):
                        i = int(min(pyramid[m, t], self.sampling_timesteps - 1))
                        x_t, z_chunk = self.transition_model.ddim_sample_step(
                            chunk[:, t], z_chunk, None, i
                        )
                        chunk[:, t] = x_t

                z = z_chunk  # carry over

            else:
                # ---------------- BiGRU sequence sampler ----------------
                for m in range(pyramid_height):
                    if self.transition_model.return_all_timesteps:
                        xs_pred_all.append([chunk[:, t].clone() for t in range(horizon)])

                    i_vec = torch.as_tensor(pyramid[m], device=self.device, dtype=torch.long)  # [horizon]

                    # single pass through the transition model in sequence mode
                    # It returns the posterior z_chunk_tm1 for conditioning next step
                    z_chunk_tm1, _, _, _ = self.transition_model(
                        None, chunk[:, :-1], None, deterministic_t=None, cum_snr=None
                    )

                    # One DDIM step on entire sequence 
                    chunk = self.transition_model.ddim_sample_step_sequence(
                        x_seq=chunk[:, -1:],           # [B, 1, Cx, H, W]
                        cond_seq=None,                 # None
                        i_vec=i_vec[-1:],              # [horizon] --> [-1] --> only last index used
                        z_cond=z_chunk_tm1[:, -1:]     # [B, 1, Cz, H, W]
                    )

            # append to predictions (time-first list)
            for t in range(horizon):
                xs_pred.append(chunk[:, t])

        # ---------------- metrics & logging ----------------
        xs_pred = torch.stack(xs_pred)  # [T, B, (fs*C), H, W]
        loss = F.mse_loss(xs_pred, xs, reduction="none")
        loss = self.reweigh_loss(loss, masks)

        xs_vis      = rearrange(xs,      "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        xs_pred_vis = rearrange(xs_pred, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)

        xs_vis      = self._unnormalize_x(xs_vis)
        xs_pred_vis = self._unnormalize_x(xs_pred_vis)

        if not self.is_spatial:
            if self.transition_model.return_all_timesteps:
                xs_pred_all = [torch.stack(item) for item in xs_pred_all]
                limit = self.transition_model.sampling_timesteps
                for i in np.linspace(1, limit, 5, dtype=int):
                    xs_pred_i = xs_pred_all[i]
                    xs_pred_i = self._unnormalize_x(xs_pred_i)
                    metric_dict = get_validation_metrics_for_states(xs_pred_i, xs_vis)
                    self.log_dict(
                        {f"{namespace}/{i}_sampling_steps_{k}": v for k, v in metric_dict.items()},
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                    )
            else:
                metric_dict = get_validation_metrics_for_states(xs_pred_vis, xs_vis)
                self.log_dict(
                    {f"{namespace}/{k}": v for k, v in metric_dict.items()},
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )

        self.validation_step_outputs.append((xs_pred_vis.detach().cpu(), xs_vis.detach().cpu()))
        return loss

    # new validation step with chunk-wise Bi-GRU sampler (Second version)
    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation"):
        if self.calc_crps_sum:
            # repeat batch for crps sum for time series prediction
            batch = [d[None].expand(self.calc_crps_sum, *([-1] * len(d.shape))).flatten(0, 1) for d in batch]

        # ---------------- preliminaries ----------------
        # get preprocessed data
        # here xs is the full ground truth sequence
        xs, conditions, masks, *_, init_z = self._preprocess_batch(batch)

        # initialize helpers
        # here n_frames is the total number of frames in the sequence
        n_frames, batch_size, *_ = xs.shape
        xs_pred = []
        xs_pred_all = []

        # ---------------- context ----------------
        # in the sliding window sampler, we set the context window for all steps
        # to have the same number of frames as the context length. For imputation,
        # in the special case of symmetric context, the context window is split in half.
        window_length = self.context_frames // self.frame_stack 
        if self.symmetric_context:
            window_length = window_length // 2
        if not self._is_bi_gru():
            # original GRU context rollout (deterministic)
            z = init_z
            for t in range(0, window_length):
                z, x_next_pred, _, _ = self.transition_model(z, xs[t], conditions[t], deterministic_t=0)
                xs_pred.append(x_next_pred)
        else:
            # BiGRU: treat context as given (teacher-forced), no running z needed
            for t in range(0, window_length):
                xs_pred.append(xs[t])
        
        # ---------------- prediction ----------------
        # For BiGRU, we switch to the sliding-window sampler.
        
        while len(xs_pred) < n_frames:

            # horizon for forward-GRU branch
            if self.chunk_size > 0:
                horizon = min(n_frames - len(xs_pred), self.chunk_size)
            else:
                horizon = n_frames - len(xs_pred)
            
             # print progress
            print(f"Validation step: predicting frame {len(xs_pred)+horizon} / {n_frames}", file=sys.stderr)
            
            # perform horizon steps of sampling
            if not self._is_bi_gru():
                # ---------------- original GRU per-frame sampler (UNCHANGED) ----------------
                # initialize chunk as random noise: [B, horizon, (fs*C), H, W]
                chunk = torch.stack(
                    [torch.randn((batch_size,) + tuple(self.x_stacked_shape), device=self.device)
                    for _ in range(horizon)],
                    dim=1
                )

                # pyramid timesteps for uncertainty-aware sampling
                pyramid_height = self.sampling_timesteps + int(horizon * self.uncertainty_scale)
                pyramid = np.zeros((pyramid_height, horizon), dtype=int)

                for m in range(pyramid_height):
                    for t in range(horizon):
                        pyramid[m, t] = m - int(t * self.uncertainty_scale)
                pyramid = np.clip(pyramid, a_min=0, a_max=self.sampling_timesteps, dtype=int)

                z_chunk = z.detach()
                for m in range(pyramid_height):
                    if self.transition_model.return_all_timesteps:
                        xs_pred_all.append([chunk[:, t].clone() for t in range(horizon)])

                    for t in range(horizon):
                        i = int(min(pyramid[m, t], self.sampling_timesteps - 1))
                        x_t, z_chunk = self.transition_model.ddim_sample_step(
                            chunk[:, t], z_chunk, None, i
                        )
                        chunk[:, t] = x_t

                z = z_chunk  # carry over

                # append to predictions (time-first list)
                for t in range(horizon):
                    xs_pred.append(chunk[:, t])
            else:
                # ---------------- BiGRU sequence sampler ---------------    
                # we need to handle the case where no context frames are available
                # in this case chunk is None (i.e., we start from pure noise)
                chunk = None
                if len(xs_pred) > 0:
                    if window_length == 0:
                        # when no context frames, we only take the last chunk_size 
                        # frames to perform windowed sampling. Here no need to use
                        # max as we already checked len(xs_pred) >= chunk_size, since
                        # when no window_length is used the first items in xs_pred
                        # are always chunk_size frames.
                        start_idx = len(xs_pred) - self.chunk_size
                    else:
                        # when context frames are used, we take the last window_length
                        # frames to perform windowed sampling. 
                        start_idx = max(0, len(xs_pred) - window_length)
                    
                    # construct chunk [B, window_length, (fs*C), H, W] or [B, chunk_size, (fs*C), H, W]
                    chunk = torch.stack(xs_pred[start_idx:], dim=1)  

                # append random noise frames to fill up to horizon
                random_frames = torch.randn(
                    (batch_size, horizon ) + tuple(self.x_stacked_shape), device=self.device
                ) # [B, horizon, (fs*C), H, W]
                chunk = torch.cat([chunk, random_frames], dim=1) if chunk is not None else random_frames  # [B, window_length, (fs*C), H, W]

                # if the validation task is imputation, we replace the last random noise
                # frame(s) with the ground truth frame(s) that we use for the backward-GRU branch
                if self.imputation_as_val:
                    if self.symmetric_context: # we leave this as the only option for now
                        future_idx_start = len(xs_pred) + horizon
                        gt_right = xs[future_idx_start: future_idx_start + window_length]
                        gt_right = torch.stack([gt_right[t] for t in range(window_length)], dim=1)  # [B, window_length, (fs*C), H, W]
                        chunk = torch.cat([chunk, gt_right], dim=1) # [B, 2*window_length + horizon, (fs*C), H, W]
                        assert chunk.size(1) == self.cfg.n_frames, \
                            "For imputation with symmetric context, the size of the chunk fed" \
                            " to the model must equal total number of frames"
                    else: # this is redundant due to previous checks (but kept as a reminder to implement later)
                        raise NotImplementedError("Asymmetric context not implemented for imputation yet")

                # pyramid timesteps for uncertainty-aware sampling
                pyramid_height = self.sampling_timesteps + int(horizon * self.uncertainty_scale)
                pyramid = np.zeros((pyramid_height, horizon), dtype=int)
                for m in range(pyramid_height):
                    for t in range(horizon):
                        pyramid[m, t] = m - int(t * self.uncertainty_scale)
                pyramid = np.clip(pyramid, a_min=0, a_max=self.sampling_timesteps, dtype=int)

                # Get the posterior latent for conditioning
                # single pass through the transition model in sequence mode
                if len(xs_pred) > 0:
                    
                    # if imputation, we whole chunk is fed to get both fwd_z and bwd_z,
                    # otherwise, only the context frames are used to get fwd_z
                    if self.imputation_as_val:
                        _, _, _, _, fwd_z, bwd_z = self.transition_model(
                            None, chunk, None, 
                            deterministic_t=0, cum_snr=None, return_z_decoupled=True
                        )
                    else:
                        _, _, _, _, fwd_z, _ = self.transition_model(
                            None, chunk[:, :window_length], None, 
                            deterministic_t=0, cum_snr=None, return_z_decoupled=True
                        )

                    # prepare conditioning latents for DDIM steps:
                    # if "imputation", both forward and backward latents are used
                    # where the forward latent comes from the last context frame
                    # and the backward latent comes from the future ground truth frame
                    # if "forecasting", only the forward latent is used
                    if self.imputation_as_val:
                        if self.symmetric_context: # only option for now 
                            z_cond = (fwd_z[:, window_length-1], bwd_z[:, window_length+horizon])
                        else: # again redundant due to previous checks but kept as a reminder
                            raise NotImplementedError("Asymmetric context not implemented for imputation yet")
                    else:
                        z_cond = (fwd_z[:, -1], None)
                        
                else:
                    fwd_z = None
                    bwd_z = None
                    z_cond = (None, None)
                

                # Run vectorized DDIM steps across the whole window
                for m in range(pyramid_height):
                    
                    if self.transition_model.return_all_timesteps:
                        xs_pred_all.append([chunk[:, t].clone() for t in range(window_length)])
                    
                    i_vec = torch.as_tensor(pyramid[m], device=self.device, dtype=torch.long)  # [horizon]
                    i_vec.clamp_(0, self.sampling_timesteps - 1)

                    if self.imputation_as_val:
                        input_seq = chunk[:, window_length:window_length + horizon]     # [B, horizon, Cx, H, W]
                    else:
                        input_seq = chunk[:, window_length:]                            # [B, horizon, Cx, H, W]
                    
                    last_chunk = self.transition_model.ddim_sample_step_sequence(
                        x_seq=input_seq,                        # [B, horizon, Cx, H, W]
                        cond_seq=None,                          # keep None unless specified otherwise
                        i_vec=i_vec,                            # [horizon], 
                        z_cond=z_cond                           # (fwd_z_cond, bwd_z_cond)
                    )
                    if self.imputation_as_val:
                        chunk[:, window_length:window_length + horizon] = last_chunk
                    else:
                        chunk[:, window_length:] = last_chunk

                # After denoising the whole window, take ONLY horizon number of frames as the next prediction
                for t in range(horizon):
                    xs_pred.append(chunk[:, window_length + t])
                
                if self.imputation_as_val:
                    # for imputation, we need add the last ground truth context frames
                    # when the total predicted frames + context frames = exceed n_frames
                    if len(xs_pred) + window_length == n_frames:
                        future_idx_start = len(xs_pred)
                        gt_right = xs[future_idx_start: future_idx_start + window_length]
                        for t in range(window_length):
                            xs_pred.append(gt_right[t])


        # ---------------- metrics & logging ----------------
        xs_pred = torch.stack(xs_pred)  # [T, B, (fs*C), H, W]
        loss = F.mse_loss(xs_pred, xs, reduction="none")
        loss = self.reweigh_loss(loss, masks)

        xs_vis      = rearrange(xs,      "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        xs_pred_vis = rearrange(xs_pred, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)

        xs_vis      = self._unnormalize_x(xs_vis)
        xs_pred_vis = self._unnormalize_x(xs_pred_vis)

        if not self.is_spatial:
            if self.transition_model.return_all_timesteps:
                xs_pred_all = [torch.stack(item) for item in xs_pred_all]
                limit = self.transition_model.sampling_timesteps
                for i in np.linspace(1, limit, 5, dtype=int):
                    xs_pred_i = xs_pred_all[i]
                    xs_pred_i = self._unnormalize_x(xs_pred_i)
                    metric_dict = get_validation_metrics_for_states(xs_pred_i, xs_vis)
                    self.log_dict(
                        {f"{namespace}/{i}_sampling_steps_{k}": v for k, v in metric_dict.items()},
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                    )
            else:
                metric_dict = get_validation_metrics_for_states(xs_pred_vis, xs_vis)
                self.log_dict(
                    {f"{namespace}/{k}": v for k, v in metric_dict.items()},
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )

        self.validation_step_outputs.append((xs_pred_vis.detach().cpu(), xs_vis.detach().cpu()))
        return loss
    #############################################################################################

    def on_validation_epoch_end(self, namespace="validation"):
        if not self.validation_step_outputs:
            return

        self.validation_step_outputs.clear()

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self.validation_step(*args, **kwargs, namespace="test")
    
    def _normalize_x(self, xs):
        if self.x_shape[-1] > 1:
            shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
            mean = self.data_mean.reshape(shape).to(xs.device)
            std = self.data_std.reshape(shape).to(xs.device)
            return (xs - mean) / std
        else:
            xs_tmp = xs.squeeze(4).squeeze(3)
            mean = self.data_mean
            std = self.data_std
            xs_tmp_n = ((xs_tmp - mean)/std).unsqueeze(3).unsqueeze(4)

            return xs_tmp_n

    def _unnormalize_x(self, xs):
        if self.x_shape[-1] > 1:
            shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
            mean = self.data_mean.reshape(shape).to(xs.device)
            std = self.data_std.reshape(shape).to(xs.device)
            return xs * std + mean
        else:
            xs_tmp = xs.squeeze(4).squeeze(3)
            mean = self.data_mean
            std = self.data_std
            xs_tmp_un = ((xs_tmp*std) + mean).unsqueeze(3).unsqueeze(4)

            return xs_tmp_un
