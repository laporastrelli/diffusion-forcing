import numpy as np
from einops import rearrange
import torch

from .df_base import DiffusionForcingBase

from algorithms.common.metrics import (
    FrechetInceptionDistance,
    LearnedPerceptualImagePatchSimilarity,
)

from utils.logging_utils import log_video, get_validation_metrics_for_videos
from pytorch_lightning.trainer.states import TrainerFn

import numpy as np
import os

import sys

class DiffusionForcingVideo(DiffusionForcingBase):
    def _build_model(self):
        super()._build_model()

        if self.cfg.compute_fid_lpips:
            self.validation_fid_model = FrechetInceptionDistance(feature=64)
            self.validation_lpips_model = LearnedPerceptualImagePatchSimilarity()
        else:
            self.validation_fid_model = None
            self.validation_lpips_model = None

        self.validation_fvd_model = None  # FrechetVideoDistance()

    def training_step(self, batch, batch_idx):
        # if batch_idx == 0:
        #     self.visualize_noise(batch)

        output_dict = super().training_step(batch, batch_idx)

        if batch_idx % 5000 == 0:
            log_video(
                output_dict["xs_pred"],
                output_dict["xs"],
                step=self.global_step,
                namespace="training_vis",
                logger=self.logger.experiment,
            )
        return output_dict

    def on_validation_epoch_end(self, namespace="validation") -> None:
        if not self.validation_step_outputs:
            return  
        
        xs_pred = []
        xs = []
        for _, (pred, gt) in enumerate(self.validation_step_outputs):                
            xs_pred.append(pred)
            xs.append(gt)
            
        xs_pred = torch.cat(xs_pred, 1)
        xs      = torch.cat(xs, 1)

        print("===============================================")
        print(xs_pred.shape, xs.shape)
        print("===============================================")

        if self.logger:
            if self.trainer.state.fn == TrainerFn.VALIDATING:
                print('###############################################################')
                xs_to_save      = xs.detach().cpu().numpy()
                xs_pred_to_save = xs_pred.detach().cpu().numpy()

                print('ground-truth shape: ', xs_to_save.shape)
                print('predicition shape : ', xs_pred_to_save.shape)

                try:
                    vae_ckpnt = self.cfg.vae_checkpoint
                except (AttributeError, KeyError):
                    vae_ckpnt = None
                if vae_ckpnt is not None:
                    root = os.path.dirname(os.path.dirname(self.cfg.vae_checkpoint))
                    save_dir = os.path.join(root, "diffusion_latents", f'validation_{self.context_length}')
                else:
                    save_dir = os.path.join(self.logger.save_dir, f'validation_{self.context_length}')
                os.makedirs(save_dir, exist_ok=True)

                print('#######################################')
                print("SAVE DIR: ", save_dir)
                print('#######################################')

                for b in range(xs_pred_to_save.shape[1]):
                    p_np = xs_pred_to_save[:, b]
                    g_np = xs_to_save[:,      b]
                    np.save(os.path.join(save_dir, 
                                         f"validation_pred_{b:05d}.npy"), p_np)
                    np.save(os.path.join(save_dir, 
                                         f"validation_gt_{b:05d}.npy"),   g_np)
                    
                print('Validation data saved successfully!')
                print('##################################################################')
                
            log_video(
                xs_pred,
                xs,
                step=None if namespace == "test" else self.global_step,
                namespace=namespace + "_vis",
                context_frames=self.context_frames,
                logger=self.logger.experiment,
            )

        metric_dict = get_validation_metrics_for_videos(
            xs_pred[self.context_frames :],
            xs[self.context_frames :],
            lpips_model=self.validation_lpips_model,
            fid_model=self.validation_fid_model,
            fvd_model=self.validation_fvd_model
        )
        self.log_dict(
            {f"{namespace}/{k}": v for k, v in metric_dict.items()}, on_step=False, on_epoch=True, prog_bar=True
        )

        self.validation_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(namespace="test")

    def visualize_noise(self, batch):
        self.log_dict({"pixel_mean": torch.mean(batch[0]), "pixel_std": torch.std(batch[0])})

        xs = self._preprocess_batch(batch)[0]

        xs = rearrange(xs, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        batch_size = xs.shape[1]
        x = xs[0]
        xs = []
        xs_noised = []
        for t in np.linspace(0, self.cfg.diffusion.timesteps - 1, 100):
            xs.append(x)
            t = torch.Tensor([int(t)] * batch_size).long().to(x.device)
            x = self.transition_model.q_sample(x, t)
            xs_noised.append(x)

        xs = self._unnormalize_x(torch.stack(xs))
        xs_noised = self._unnormalize_x(torch.stack(xs_noised))

        log_video(
            xs_noised,
            xs,
            step=self.global_step,
            namespace="noise_visualization",
            context_frames=0,
            logger=self.logger.experiment,
        )
