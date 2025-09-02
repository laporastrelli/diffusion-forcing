import torch
import os
import json

from torch.utils.data import Dataset
from omegaconf import DictConfig
from video_vae.dataset import Pendulum       
from video_vae.VAE import VAE

class PendulumLatentsDataset(Dataset):
    def __init__(self, cfg: DictConfig, split: str):

        self.cfg = cfg

        # extract useful config parameters from base config file
        self.base_root_dir = os.path.dirname(os.path.dirname(cfg.vae_checkpoint))
        self.base_config_path = os.path.join(self.base_root_dir, "config.json")
        with open(self.base_config_path, "r", encoding="utf-8") as f:
            self.base_config = json.load(f)

        # wrap the existing Pendulum dataset (handles grouping & loading)
        self.base       = Pendulum(root_dir=cfg.save_dir, 
                                   split="train" if split=="training" else "test", 
                                   group=self.base_config["group"])
        self.frame_skip = cfg.frame_skip

        # load one sample from base datase to get data specs
        tmp_sample    = self.base[1]
        n_channels_in = tmp_sample.shape[1]
        image_size    = tmp_sample.shape[-1]

        # load & freeze your pretrained VAE
        vae = VAE(n_channels_in=n_channels_in, 
                  image_size=image_size, 
                  n_channels=self.base_config["n_channels"])
        vae.load_state_dict(torch.load(cfg.vae_checkpoint, map_location="cpu"))
        self.vae = vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # get grouped video: [T', C_grouped, H, W]
        video    = self.base[idx]

        # make nonterminal flags & apply frame-skip
        nonterm  = torch.ones(video.shape[0], dtype=torch.float32)
        video    = video[::self.frame_skip]
        nonterm  = nonterm[::self.frame_skip]

        # encode *all* frames at once: VAE treats T' as batch dim
        with torch.no_grad():
            latents = self.vae.loss_function(video, latents_only=True)  # → [T', C_lat, H_lat, W_lat]

        # return (latents, nonterm, idx)
        return latents, nonterm, idx

    def get_data_lengths(self):
        # length in tokens = (original_frames // group) // frame_skip
        base_lens = self.base.get_data_lengths()
        return [(L // self.base.group) // self.frame_skip for L in base_lens]
