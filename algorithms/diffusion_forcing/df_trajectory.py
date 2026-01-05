from omegaconf import DictConfig
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from algorithms.common.metrics import (
    FrechetInceptionDistance,
    LearnedPerceptualImagePatchSimilarity,
    FrechetVideoDistance,
)
from .df_base import DiffusionForcingBase
from utils.logging_utils import log_video, get_validation_metrics_for_videos
from pytorch_lightning.trainer.states import TrainerFn

import numpy as np
import os

import sys

class DiffusionForcingTrajectory(DiffusionForcingBase):
    """
    A trajectory prediction algorithm using Diffusion Forcing.
    """

    def __init__(self, cfg: DictConfig):
        self.metrics = cfg.metrics
        self.n_tokens = cfg.n_frames // cfg.frame_stack  # number of max tokens for the model
        self.context_length = cfg.context_frames
        self.cfg = cfg
        super().__init__(cfg)

    def _build_model(self):
        super()._build_model()
        self.validation_fid_model = FrechetInceptionDistance(feature=64) if "fid" in self.metrics else None
        self.validation_lpips_model = LearnedPerceptualImagePatchSimilarity() if "lpips" in self.metrics else None
        self.validation_fvd_model = [FrechetVideoDistance()] if "fvd" in self.metrics else None

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        output_dict = super().training_step(batch, batch_idx)
        # log the trajectory video
        if batch_idx % 5000 == 0 and self.logger:
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
        idxs = []
        for i, (pred, gt, idxs_batch) in enumerate(self.validation_step_outputs):                
            xs_pred.append(pred)
            xs.append(gt)
            if idxs_batch is None:
                tmp = np.arange(i*pred.size(1), i*pred.size(1) + pred.size(1))
                tmp_torch = torch.from_numpy(tmp)
                idxs.append(tmp_torch)
            else:
                idxs.append(idxs_batch)

        xs_pred = torch.cat(xs_pred, 1)
        xs      = torch.cat(xs, 1)
        idxs    = torch.cat(idxs, 0)

        print("===============================================")
        print(xs_pred.shape, xs.shape, idxs.shape)
        print("===============================================")

        if self.logger:
            if self.trainer.state.fn == TrainerFn.VALIDATING:
                print('###############################################################')
                xs_to_save      = xs.detach().cpu().numpy()
                xs_pred_to_save = xs_pred.detach().cpu().numpy()
                idxs_to_save    = idxs.detach().cpu().numpy()

                print('ground-truth shape: ', xs_to_save.shape)
                print('predicition shape : ', xs_pred_to_save.shape)
                print('indexes shape     : ', idxs_to_save.shape)

                try:
                    vae_ckpnt = self.cfg.vae_checkpoint
                except (AttributeError, KeyError):
                    vae_ckpnt = None
                except Exception as e:
                    print(f"caught {type(e).__name__}: {e}")
                    print("--------------------------------")
                    print("WARNING: no vae checkpoint found")
                    print("--------------------------------")
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

                for b, vid_idx in enumerate(idxs):
                    p_np = xs_pred_to_save[:, b]
                    g_np = xs_to_save[:,      b]
                    np.save(os.path.join(save_dir, 
                                         f"validation_pred_{vid_idx:05d}.npy"), p_np)
                    np.save(os.path.join(save_dir, 
                                         f"validation_gt_{vid_idx:05d}.npy"),   g_np)
                    
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
            fvd_model=(self.validation_fvd_model[0] if self.validation_fvd_model else None),
            metrics=self.metrics,
        )
        self.log_dict(
            {f"{namespace}/{k}": v for k, v in metric_dict.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.validation_step_outputs.clear()
