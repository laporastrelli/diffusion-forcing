import torch
import numpy as np

from omegaconf import DictConfig
from .base_video_dataset import BaseVideoDataset

from pathlib import Path

import sys


class SinglePendulumFullDataset(BaseVideoDataset):
    '''
    Bouncing Balls Dataset
    '''
    def __init__(self, cfg: DictConfig, split: str = "training"):
        if split == "test":
            split = "validation"
        super().__init__(cfg, split)
    
    def download_dataset(self):
        pass
    
    def get_data_paths(self, split):
        data_dir = self.save_dir / split
        paths = sorted(list(data_dir.glob("**/*.npz")), key=lambda x: x.name)
        return paths
    
    def load_video(path):
        video = np.load(path)['latents']
        return video
    
    def get_data_lengths(self, split):
        if split == 'training':
            lengths = [60] * len(self.get_data_paths(split))
        else:
            lengths = [360] * len(self.get_data_paths(split))
        return lengths

    def __getitem__(self, idx):
        video_path = self.data_paths[idx]
        video = np.load(video_path, allow_pickle=True)['latents']
        
        pad_len = self.n_frames - len(video)
        assert pad_len == 0, "There should be no padding to be added"

        nonterminal = np.ones(self.n_frames)
        if len(video) < self.n_frames:
            video = np.pad(video, ((0, pad_len), (0, 0), (0, 0), (0, 0)))
            nonterminal[-pad_len:] = 0

        video = torch.from_numpy(video / 256.0).float().permute(0, 3, 1, 2).contiguous()

        if self.external_cond_dim:
            external_cond = np.load(
                # pylint: disable=no-member
                self.condition_dir
                / f"{video_path.name.replace('.mp4', '.npy')}"
            )
            if len(external_cond) < self.n_frames:
                external_cond = np.pad(external_cond, ((0, pad_len),))
            external_cond = torch.from_numpy(external_cond).float()
            return (
                video[:: self.frame_skip],
                external_cond[:: self.frame_skip],
                nonterminal[:: self.frame_skip],
            )
        else:
            return video[:: self.frame_skip], nonterminal[:: self.frame_skip]

