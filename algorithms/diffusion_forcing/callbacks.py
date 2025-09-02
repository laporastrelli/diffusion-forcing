# algorithms/diffusion_forcing/callbacks.py

import os
import numpy as np
import torch
from lightning.pytorch.callbacks.callback import Callback

class SaveTrainReconsCallback(Callback):
    """
    After .fit() finishes (all epochs + final validation),
    use the provided train_loader to save gt-recon pairs
    under logger.save_dir/<subfolder>.
    """

    def __init__(self, 
                 train_loader,
                 dataset_config, 
                 subfolder: str = "training"):
        super().__init__()
        self.train_loader = train_loader
        self.dataset_cfg = dataset_config
        self.subfolder = subfolder

    def on_fit_end(self, trainer, pl_module):
        inputs_all, recons_all, idxs_all = [], [], []
        device = pl_module.device

        # mute logging inside training_step
        orig_log = pl_module.log
        pl_module.log = lambda *args, **kwargs: None

        pl_module.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.train_loader):
                out = pl_module.training_step(batch, batch_idx)
                xs      = out["xs"]
                xs_pred = out["xs_pred"]
                idxs    = out["idxs"]

                inputs_all.append(xs.cpu().numpy())
                recons_all.append(xs_pred.cpu().numpy())
                idxs_all.append(idxs.cpu().numpy())

        pl_module.train()

        # restore original log method
        pl_module.log = orig_log

        # stack and save
        inputs_all = np.concatenate(inputs_all, axis=1)
        recons_all = np.concatenate(recons_all, axis=1)
        idxs_all   = np.concatenate(idxs_all, axis=0)

        # derive your logger’s save_dir
        if self.dataset_cfg is None:
            root = getattr(pl_module.logger, "save_dir", os.getcwd())
        else:  
            root = os.path.dirname(os.path.dirname(self.dataset_cfg.vae_checkpoint))
        save_dir = os.path.join(root, "diffusion_latents", self.subfolder)
        os.makedirs(save_dir, exist_ok=True)

        # save each gt-recon pair individually
        for b, vid_idx in enumerate(idxs_all):
            p_np = recons_all[:, b]
            g_np = inputs_all[:, b]
            np.save(os.path.join(save_dir, f"train_recon_{vid_idx:05d}.npy"), p_np)
            np.save(os.path.join(save_dir, f"train_gt_{vid_idx:05d}.npy"),    g_np)

        print(f"✅ Saved training ground-truth and reconstructions into {save_dir}")
