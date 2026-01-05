import torch
import numpy as np

from omegaconf import DictConfig
from .base_video_dataset import BaseVideoDataset

import sys

class SinglePendulumTrajDataset(BaseVideoDataset):

    '''
    SinglePendulumTrajDataset
    '''
    def __init__(self, cfg: DictConfig, split: str = "training"):
        if split == "test":
            split = "validation"
        super().__init__(cfg, split)
        self.use_augmented_data = cfg.get("use_augmented_data", False)
    
    def download_dataset(self):
        pass

    def get_data_paths(self, split):
        data_dir = self.save_dir / split
        paths = sorted(list(data_dir.glob("**/*.npz")), key=lambda x: x.name)
        return paths
    
    def get_data_lengths(self, split):
        if split == 'training':
            lengths = [60] * len(self.get_data_paths(split))
        else:
            paths = self.get_data_paths(split)
            test_length = np.load(paths[0])['series'].shape[0]
            lengths = [test_length] * len(self.get_data_paths(split))
        return lengths

    def __getitem__(self, idx):
        video_path = self.data_paths[idx]
        video = np.load(video_path, allow_pickle=True)['series']
        video = torch.from_numpy(video).float()
        if self.use_augmented_data:
            video = video.unsqueeze(-1).unsqueeze(-1)  # shape (T, D, 1, 1)

        pad_len = self.n_frames - len(video)
        assert pad_len == 0, "There should be no padding to be added"

        nonterminal = np.ones(self.n_frames)

        return video, nonterminal