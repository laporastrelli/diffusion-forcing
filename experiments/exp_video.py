from datasets.video import (
    MinecraftVideoDataset,
    DmlabVideoDataset,
    BouncingBallsDataset,
    BouncingBallsTrajDataset,
    PendulumLatentsDataset,
    SinglePendulumFullDataset,
    DoublePendulumDataset,
    TwoBodyDataset
)
from algorithms.diffusion_forcing import DiffusionForcingVideo
from .exp_base import BaseLightningExperiment


class VideoPredictionExperiment(BaseLightningExperiment):
    """
    A video prediction experiment
    """

    compatible_algorithms = dict(
        df_video=DiffusionForcingVideo,
    )

    compatible_datasets = dict(
        # video datasets
        video_minecraft=MinecraftVideoDataset,
        video_dmlab=DmlabVideoDataset,
        video_bouncing_balls=BouncingBallsDataset,
        video_bouncing_balls_traj=BouncingBallsTrajDataset,
        video_single_pendulum_full_vae_latents=PendulumLatentsDataset,
        video_single_pendulum_full=SinglePendulumFullDataset,
        video_double_pendulum=DoublePendulumDataset,
        video_double_pendulum_vae_latents=PendulumLatentsDataset,
        video_two_body=TwoBodyDataset
    )
