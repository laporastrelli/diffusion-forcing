from datasets.video import (
    SinglePendulumTrajDataset
)
from algorithms.diffusion_forcing import DiffusionForcingTrajectory
from .exp_base import BaseLightningExperiment


class TrajectoryPredictionExperiment(BaseLightningExperiment):
    """
    A trajectory prediction experiment
    """

    compatible_algorithms = dict(
        df_trajectory=DiffusionForcingTrajectory,
    )

    compatible_datasets = dict(
        # trajectory datasets
        trajectory_single_pendulum=SinglePendulumTrajDataset
    )
