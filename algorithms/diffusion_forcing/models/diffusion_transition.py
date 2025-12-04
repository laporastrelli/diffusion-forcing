from typing import Optional, Tuple, Union

from random import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

from .unet import TransitionUnet, TransitionMlp, BiGRUTransitionUNet
from .resnet import ResBlock2d
from .utils import extract, default, linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule


ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start", "pred_z", "model_out", "fwd_z", "bwd_z"])


class DiffusionTransitionModel(nn.Module):
    def __init__(self, x_shape, z_shape, external_cond_dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.x_shape = x_shape
        self.z_shape = z_shape
        self.external_cond_dim = external_cond_dim
        self.mask_unet = cfg.mask_unet
        self.num_gru_layers = cfg.num_gru_layers
        self.num_mlp_layers = cfg.num_mlp_layers
        self.timesteps = cfg.timesteps
        self.sampling_timesteps = cfg.sampling_timesteps
        self.beta_schedule = cfg.beta_schedule
        self.objective = cfg.objective
        self.use_snr = cfg.use_snr
        self.use_cum_snr = cfg.use_cum_snr
        self.snr_clip = cfg.snr_clip
        self.cum_snr_decay = cfg.cum_snr_decay
        self.ddim_sampling_eta = cfg.ddim_sampling_eta
        self.clip_noise = cfg.clip_noise
        self.p2_loss_weight_gamma = cfg.p2_loss_weight_gamma
        self.p2_loss_weight_k = cfg.p2_loss_weight_k
        self.schedule_fn_kwargs = cfg.schedule_fn_kwargs
        self.self_condition = cfg.self_condition
        self.network_size = cfg.network_size
        self.return_all_timesteps = cfg.return_all_timesteps

        if self.objective not in ["pred_noise", "pred_x0", "pred_v"]:
            raise ValueError("objective must be either pred_noise or pred_x0 or pred_v ")

        self._build_model()
        self._build_buffer()

    def _build_model(self):
        x_channel = self.x_shape[0]
        z_channel = self.z_shape[0]
        if len(self.x_shape) == 3:

            if self.cfg.transition_type == "bi-gru":
                self.model = BiGRUTransitionUNet(
                    z_channel=z_channel,
                    x_channel=x_channel,
                    z_shape=self.z_shape,
                    external_cond_dim=self.external_cond_dim,
                    network_size=self.network_size,
                    num_gru_layers=self.num_gru_layers,
                    self_condition=self.self_condition,
                )

            else:
                self.model = TransitionUnet(
                    z_channel=z_channel,
                    x_channel=x_channel,
                    external_cond_dim=self.external_cond_dim,
                    network_size=self.network_size,
                    num_gru_layers=self.num_gru_layers,
                    self_condition=self.self_condition,
                )

            self.x_from_z = nn.Sequential(
                ResBlock2d(z_channel, x_channel),
                nn.Conv2d(x_channel, x_channel, 1, padding=0),
            )

        elif len(self.x_shape) == 1:
            self.model = TransitionMlp(
                z_dim=z_channel,
                x_dim=x_channel,
                external_cond_dim=self.external_cond_dim,
                network_size=self.network_size,
                num_gru_layers=self.num_gru_layers,
                num_mlp_layers=self.num_mlp_layers,
                self_condition=self.self_condition,
            )

            self.x_from_z = nn.Linear(z_channel, x_channel)

        else:
            raise ValueError(f"x_shape must have 1 or 3 dims but got shape {self.x_shape}")

    def _build_buffer(self):
        if self.beta_schedule == "linear":
            beta_schedule_fn = linear_beta_schedule
        elif self.beta_schedule == "cosine":
            beta_schedule_fn = cosine_beta_schedule
        elif self.beta_schedule == "sigmoid":
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f"unknown beta schedule {self.beta_schedule}")

        betas = beta_schedule_fn(self.timesteps, **self.schedule_fn_kwargs)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer(
            "posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        # calculate p2 reweighting

        register_buffer(
            "p2_loss_weight",
            (self.p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -self.p2_loss_weight_gamma,
        )

        # derive loss weight
        # https://arxiv.org/abs/2303.09556
        # snr: signal noise ratio
        snr = alphas_cumprod / (1 - alphas_cumprod)
        clipped_snr = snr.clone()
        clipped_snr.clamp_(max=self.snr_clip)

        register_buffer("clipped_snr", clipped_snr)
        register_buffer("snr", snr)

    def rollout(
        self,
        z: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param z: z at current time step
        :param external_cond: external_cond to be conditioned on
        :return: z_next_pred, x_next_pred:
        """
        batch_size = z.shape[0]

        # run a full diffusion to predict next latent z and obs x
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        x_next_pred, z_next_pred = sample_fn(
            (batch_size,) + tuple(self.x_shape),
            z,
            external_cond=external_cond,
            return_all_timesteps=self.return_all_timesteps,
        )
        return z_next_pred, x_next_pred

    def forward(
        self,
        z: torch.Tensor,
        x_next: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
        deterministic_t: Optional[Union[float, int]] = None,
        cum_snr: Optional[torch.Tensor] = None,
        return_z_decoupled: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param z: z at current time step
        :param x_next: ground truth x x at next step
        :param external_cond: external_cond to be conditioned on
        :param deterministic_t: set a noise level t directly instead of sampling.
        :param cum_snr: cumulative snr for previous time steps

        :return: z_next_pred, x_next_pred, loss(unreduced), cum_snr
        """
        if z is not None and z.dim() == 4:
            batch_size = z.shape[0]
            if deterministic_t is None:
                t = torch.randint(0, self.num_timesteps, (batch_size,), device=z.device).long()
            elif isinstance(deterministic_t, float):
                deterministic_t = round(deterministic_t * (self.num_timesteps - 1))
                t = torch.full((batch_size,), deterministic_t, device=z.device).long()
            elif isinstance(deterministic_t, int):
                deterministic_t = deterministic_t if deterministic_t >= 0 else self.timesteps + deterministic_t
                t = torch.full((batch_size,), deterministic_t, device=z.device).long()

            # get noised version of x_next
            noise = torch.randn_like(x_next)
            noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
            noised_x_next = self.q_sample(x_start=x_next, t=t, noise=noise)

            x_self_cond = None
            if self.self_condition and random() < 0.5:
                with torch.no_grad():
                    x_self_cond = self.model_predictions(noised_x_next, t, z, external_cond=external_cond).pred_x_start
                    x_self_cond.detach_()

            model_pred = self.model_predictions(noised_x_next, t, z, external_cond=external_cond, x_self_cond=x_self_cond)
            x_next_pred = model_pred.pred_x_start
            z_next_pred = model_pred.pred_z

            pred = model_pred.model_out

            if self.objective == "pred_noise":
                target = noise
            elif self.objective == "pred_x0":
                target = x_next
            elif self.objective == "pred_v":
                target = self.predict_v(x_next, t, noise)
            else:
                raise ValueError(f"unknown objective {self.objective}")

            loss = F.mse_loss(pred, target.detach(), reduction="none")

            normalized_clipped_snr = self.clipped_snr[t] / self.snr_clip
            normalized_snr = self.snr[t] / self.snr_clip

            if cum_snr is None or not self.use_cum_snr:
                cum_snr_next = normalized_clipped_snr
                clipped_fused_snr = normalized_clipped_snr
                fused_snr = normalized_snr
            else:
                cum_snr_next = cum_snr * self.cum_snr_decay + normalized_clipped_snr * (1 - self.cum_snr_decay)
                clipped_fused_snr = 1 - (1 - cum_snr * self.cum_snr_decay) * (1 - normalized_clipped_snr)
                fused_snr = 1 - (1 - cum_snr * self.cum_snr_decay) * (1 - normalized_snr)

            if self.use_snr:
                if self.objective == "pred_noise":
                    loss_weight = clipped_fused_snr / fused_snr
                elif self.objective == "pred_x0":
                    loss_weight = clipped_fused_snr * self.snr_clip
                elif self.objective == "pred_v":
                    loss_weight = clipped_fused_snr * self.snr_clip / (fused_snr * self.snr_clip + 1)
                loss_weight = loss_weight.view(batch_size, *((1,) * (len(loss.shape) - 1)))
            elif self.use_cum_snr and cum_snr is not None:
                loss_weight = cum_snr * self.snr_clip
                loss_weight = loss_weight.view(batch_size, *((1,) * (len(loss.shape) - 1)))
            else:
                loss_weight = torch.ones_like(loss)
                loss_weight *= self.snr_clip * 0.5  # multiply by a constant so weight scale is similar to snr

            loss = loss * loss_weight

            return z_next_pred, x_next_pred, loss, cum_snr_next
        
        elif z is None or z.dim() == 5:
            # z is set to None during training since bi-gru handles initial states internally
            assert (z is None ) and x_next.dim() == 5, "z must be None and x_next must be 5D when using sequence mode"
            B, T = x_next.shape[:2]

            # sample per-frame t and noise (flatten for reuse of existing helpers)
            x_f      = x_next.reshape(B*T, *x_next.shape[2:])
            t_f = None
            if deterministic_t is None:
                t_f = torch.randint(0, self.num_timesteps, (B*T,), device=x_next.device).long()
            elif isinstance(deterministic_t, float):
                d = round(deterministic_t * (self.num_timesteps - 1))
                t_f = torch.full((B*T,), d, device=x_next.device).long()
            elif isinstance(deterministic_t, int):
                d = deterministic_t if deterministic_t >= 0 else self.timesteps + deterministic_t
                t_f = torch.full((B*T,), d, device=x_next.device).long()

            # get noised version of x_next
            noise_f = torch.randn_like(x_f).clamp_(-self.clip_noise, self.clip_noise)
            noised_x_f = self.q_sample(x_start=x_f, t=t_f, noise=noise_f)

            # reshape back to sequence to feed the BiGRU UNet
            noised_x = noised_x_f.reshape(B, T, *x_next.shape[2:])
            z_seq    = z.reshape(B, T, *z.shape[2:]) if z is not None else None
            t_seq    = t_f.reshape(B, T)

            # (optional) self-conditioning per-frame
            x_self_cond = None
            if self.self_condition and random() < 0.5:
                with torch.no_grad():
                    mp = self.model_predictions(noised_x, t_seq, z_seq, external_cond=external_cond)
                    x_self_cond = mp.pred_x_start
                    x_self_cond.detach_()

            # get model predictions
            mp = self.model_predictions(noised_x, t_seq, z_seq, external_cond=external_cond, x_self_cond=x_self_cond)
            x_next_pred = mp.pred_x_start          # [B,T,Cx,H,W]
            z_next_pred = mp.pred_z                # [B,T,Cz,H,W]
            pred        = mp.model_out             # [B,T,Cx,H,W] if pred_noise/x0/v
            fwd_z       = mp.fwd_z                 # [B,T,Cz,H,W]
            bwd_z       = mp.bwd_z                 # [B,T,Cz,H,W]

            # get target per-frame
            target = (
                noise_f.reshape(B, T, *x_next.shape[2:]) if self.objective == "pred_noise"
                else x_next if self.objective == "pred_x0"
                else self.predict_v(x_next.reshape(B*T, *x_next.shape[2:]), t_f, noise_f).reshape(B, T, *x_next.shape[2:])
            )

            # compute per-frame loss
            loss = F.mse_loss(pred, target.detach(), reduction="none")

            # SNR weights per-frame (reuse same formulas; just broadcast over T)
            norm_clip = self.clipped_snr[t_f] / self.snr_clip
            norm = self.snr[t_f] / self.snr_clip

            if cum_snr is None or not self.use_cum_snr:
                cum_snr_next = norm_clip.view(B, T)   # keep a per-frame cum_snr if you need it
                clipped_fused_snr = norm_clip
                fused_snr = norm
            else:
                cum_flat = cum_snr.view(B*T)
                cum_snr_next_flat = cum_flat * self.cum_snr_decay + norm_clip * (1 - self.cum_snr_decay)
                cum_snr_next = cum_snr_next_flat.view(B, T)
                clipped_fused_snr = 1 - (1 - cum_flat * self.cum_snr_decay) * (1 - norm_clip)
                fused_snr = 1 - (1 - cum_flat * self.cum_snr_decay) * (1 - norm)

            if self.use_snr:
                if self.objective == "pred_noise":
                    loss_weight = clipped_fused_snr / fused_snr
                elif self.objective == "pred_x0":
                    loss_weight = clipped_fused_snr * self.snr_clip
                elif self.objective == "pred_v":
                    loss_weight = clipped_fused_snr * self.snr_clip / (fused_snr * self.snr_clip + 1)
                loss_weight = loss_weight.view(B, T, *([1] * (loss.dim() - 2)))
            elif self.use_cum_snr and cum_snr is not None:
                loss_weight = cum_snr.view(B, T) * self.snr_clip
                loss_weight = loss_weight.view(B, T, *([1] * (loss.dim() - 2)))
            else:
                loss_weight = torch.ones_like(loss)
                loss_weight *= self.snr_clip * 0.5

            loss = loss * loss_weight
            if return_z_decoupled:
                return z_next_pred, x_next_pred, loss, cum_snr_next, fwd_z, bwd_z
            return z_next_pred, x_next_pred, loss, cum_snr_next
        
        else:
            raise ValueError(f"z must have 4 or 5 dims but got shape {z.shape}")

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, z_cond, external_cond=None, x_self_cond=None):
        fwd_z, bwd_z, z_next = self.model(x, t, z_cond, external_cond, x_self_cond)
        if z_next.dim() == 5:
            model_output = self.x_from_z(z_next.view(-1, *z_next.shape[2:]))  # flatten for conv/linear
            model_output = model_output.view(x.shape)
        else:
            model_output = self.x_from_z(z_next)

        if self.objective == "pred_noise":
            pred_noise = torch.clamp(model_output, -self.clip_noise, self.clip_noise)
            x_start = self.predict_start_from_noise(x, t, pred_noise)

        elif self.objective == "pred_x0":
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start, z_next, model_output, fwd_z, bwd_z)

    def p_mean_variance(self, x, t, z_cond, external_cond=None, x_self_cond=None):
        model_pred = self.model_predictions(x, t, z_cond, external_cond=external_cond, x_self_cond=x_self_cond)
        x_start = model_pred.pred_x_start
        pred_z = model_pred.pred_z

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start, pred_z

    # @torch.no_grad()
    def p_sample(self, x, t, z_cond, external_cond=None, x_self_cond=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start, pred_z = self.p_mean_variance(
            x, batched_times, z_cond, external_cond=external_cond, x_self_cond=x_self_cond
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        pred_x = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_x, x_start, pred_z

    # @torch.no_grad()
    def p_sample_loop(self, shape, z_cond, external_cond=None, return_all_timesteps=False):
        batch, device = shape[0], self.betas.device

        x = torch.randn(shape, device=device)
        x = torch.clamp(x, -self.clip_noise, self.clip_noise)
        xs = [x]

        x_start = None

        for t in reversed(range(0, self.num_timesteps)):
            self_cond = x_start if self.self_condition else None
            x, x_start, pred_z = self.p_sample(x, t, z_cond, external_cond, x_self_cond=self_cond)
            xs.append(x)

        ret = x if not return_all_timesteps else xs

        return ret, pred_z

    # @torch.no_grad()
    def ddim_sample(self, shape, z_cond, external_cond=None, return_all_timesteps=False):
        batch, device, total_timesteps, sampling_timesteps, eta = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
        )

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device=device)
        x = torch.clamp(x, -self.clip_noise, self.clip_noise)
        xs = [x]

        x_start = None

        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            model_pred = self.model_predictions(
                x, time_cond, z_cond, external_cond=external_cond, x_self_cond=self_cond
            )
            pred_noise, x_start, pred_z, _ = model_pred

            if time_next < 0:
                x = x_start
                xs.append(x)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(x)
            noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

            x = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            xs.append(x)

        ret = x if not return_all_timesteps else xs

        return ret, pred_z

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def ddim_sample_step(
        self, x, z_cond, external_cond=None, index=0, return_x_start=False, return_guidance_const=False
    ):
        # safeguard for the first step
        if index == 0:
            x = torch.clamp(x, -self.clip_noise, self.clip_noise)

        # assign vars
        batch, device, total_timesteps, sampling_timesteps, eta = (
            x.shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
        )

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = x.to(device)

        time, time_next = time_pairs[index]
        time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
        self_cond = None
        model_pred = self.model_predictions(x, time_cond, z_cond, external_cond=external_cond, x_self_cond=self_cond)
        pred_noise = model_pred.pred_noise
        x_start = model_pred.pred_x_start
        pred_z = model_pred.pred_z

        guidance_scale = 0
        if time_next < 0:
            x = x_start
        else:
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()
            guidance_scale = (1 - alpha) - c * (1 - alpha).sqrt()

            noise = torch.randn_like(x)
            noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

            x = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
        result = [x, pred_z]
        if return_x_start:
            result.append(x_start)
        if return_guidance_const:
            result.append(guidance_scale)
        return result

    @torch.no_grad()
    def ddim_sample_step_sequence(self, x_seq, cond_seq, i_vec, z_cond=None):
        """
        BiGRU-only: one DDIM step on an entire sequence at per-time indices i_vec.

        Args:
        x_seq:    [B, T, Cx, H, W]   current noisy sample at this schedule index
        cond_seq: [B, T, D] or None
        i_vec:    [T] long           per-time *index* into the DDIM time_pairs
        z_cond:   ([B, Cz, H, W], [B, Cz, H, W]) or None
        
        Returns:
        x_seq_next: [B, T, Cx, H, W]
        """
        B, T = x_seq.shape[:2]
        device = x_seq.device

        # Build the same 'time_pairs' used by scalar DDIM
        total_timesteps = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        eta = self.ddim_sampling_eta

        times = torch.linspace(
            -1, total_timesteps - 1,
            steps=sampling_timesteps + 1,
            device=device, dtype=torch.long
        )
        times = torch.flip(times, dims=(0,))                           # reversed
        time_pairs = torch.stack([times[:-1], times[1:]], dim=1)       # [sampling_timesteps, 2]

        # Current time and "next" time for each position in the sequence
        #   time[t], time_next[t] are ints in [-1, T-1]
        time      = time_pairs[i_vec, 0]                                # [T]
        time_next = time_pairs[i_vec, 1]                                # [T]

        # Clamp inputs where this is the first schedule index 
        mask_first = (i_vec == 0).view(1, T, 1, 1, 1)
        if mask_first.any():
            x_seq = torch.where(
                mask_first,
                x_seq.clamp(-self.clip_noise, self.clip_noise),
                x_seq
            )

        # Model predictions at the *current* times
        t_seq = time.view(1, T).expand(B, T)                            # [B,T]
        mp = self.model_predictions(x_seq, t_seq, z_cond=z_cond, external_cond=cond_seq)
        pred_noise = mp.pred_noise                                      # [B,T,C,H,W]
        x_start    = mp.pred_x_start                                    # [B,T,C,H,W]

        # Vectorized DDIM update across time
        x_seq_next = self._ddim_step_from_preds(
            x_seq, pred_noise, x_start, time, time_next, eta
        )
        return x_seq_next
    
    def _ddim_step_from_preds(self, x, pred_noise, x_start, time, time_next, eta: float):
        """
        Vectorized DDIM update across the sequence.

        Args:
        x:          [B,T,C,H,W]  current sample at 'time'
        pred_noise: [B,T,C,H,W]  ε_θ(x_t, t)
        x_start:    [B,T,C,H,W]  predicted x0
        time:       [T] long     current times (like 'time' in scalar code)
        time_next:  [T] long     next times (like 'time_next' in scalar)
        eta:        float        DDIM eta
        Returns:
        x_next:     [B,T,C,H,W]
        """
        B, T = x.shape[:2]
        device = x.device
        dtype  = x.dtype

        # Mask for positions that still have a next step
        has_next = (time_next >= 0)                                     # [T]

        # Pull alphas for current/next times (only needed where has_next)
        # Avoid indexing with -1: compute only for valid positions.
        alpha      = torch.zeros(T, device=device, dtype=dtype)
        alpha_next = torch.zeros(T, device=device, dtype=dtype)
        if has_next.any():
            idx = has_next.nonzero(as_tuple=False).squeeze(-1)
            alpha[idx]      = self.alphas_cumprod[time[idx]].to(dtype)          # α_t
            alpha_next[idx] = self.alphas_cumprod[time_next[idx]].to(dtype)     # α_{t-1}

        # Compute DDIM coefficients (vectorized over T)
        # sigma = eta * sqrt((1 - α/α_next) * (1 - α_next) / (1 - α))
        sigma = torch.zeros(T, device=device, dtype=dtype)
        c     = torch.zeros(T, device=device, dtype=dtype)                       # c = sqrt(1 - α_next - sigma^2)
        sqrt_alpha_next = torch.zeros(T, device=device, dtype=dtype)

        if has_next.any():
            a  = alpha[has_next]
            an = alpha_next[has_next]
            sigma_v = eta * torch.sqrt((1 - a / an) * (1 - an) / (1 - a))
            c_v     = torch.sqrt(1 - an - sigma_v**2)
            sigma[has_next] = sigma_v
            c[has_next]     = c_v
            sqrt_alpha_next[has_next] = torch.sqrt(an)

        # Broadcast to [B,T,1,1,1]
        def _bcast(v):
            return v.view(1, T, 1, 1, 1).expand(B, T, 1, 1, 1)

        sigma_b          = _bcast(sigma)
        c_b              = _bcast(c)
        sqrt_alpha_next_b= _bcast(sqrt_alpha_next)
        has_next_b       = _bcast(has_next.to(dtype))                  # for masking

        # Sample noise where needed
        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        noise = noise * has_next_b                                     # zero where there's no next step

        # DDIM update:
        # if time_next < 0: x <- x_start
        # else: x <- sqrt(α_next)*x_start + c*pred_noise + σ*noise
        x_updated = (x_start * sqrt_alpha_next_b) + (c_b * pred_noise) + (sigma_b * noise)
        x_next    = torch.where(has_next_b.bool(), x_updated, x_start)

        return x_next

