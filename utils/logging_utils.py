from typing import Optional
import wandb
import numpy as np
import torch

import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import matplotlib.animation as animation
from pathlib import Path

from matplotlib.animation import FuncAnimation
import os
import tempfile
import matplotlib.pyplot as plt

import sys

plt.set_loglevel("warning")

from torchmetrics.functional import mean_squared_error, peak_signal_noise_ratio
from torchmetrics.functional import (
    structural_similarity_index_measure,
    universal_image_quality_index,
)
from algorithms.common.metrics import (
    FrechetVideoDistance,
    LearnedPerceptualImagePatchSimilarity,
    FrechetInceptionDistance,
)


# FIXME: clean up & check this util
def log_video(
    observation_hat,
    observation_gt=None,
    step=0,
    namespace="train",
    prefix="video",
    context_frames=0,
    color=(255, 0, 0),
    logger=None,
):
    """
    take in video tensors in range [-1, 1] and log into wandb

    :param observation_hat: predicted observation tensor of shape (frame, batch, channel, height, width)
    :param observation_gt: ground-truth observation tensor of shape (frame, batch, channel, height, width)
    :param step: an int indicating the step number
    :param namespace: a string specify a name space this video logging falls under, e.g. train, val
    :param prefix: a string specify a prefix for the video name
    :param context_frames: an int indicating how many frames in observation_hat are ground truth given as context
    :param color: a tuple of 3 numbers specifying the color of the border for ground truth frames
    :param logger: optional logger to use. use global wandb if not specified
    """

    if observation_hat.size(2) == 3:
        if not logger:
            logger = wandb
        if observation_gt is None:
            observation_gt = torch.zeros_like(observation_hat)
        observation_hat[:context_frames] = observation_gt[:context_frames]
        # Add red border of 1 pixel width to the context frames
        for i, c in enumerate(color):
            c = c / 255.0
            observation_hat[:context_frames, :, i, [0, -1], :] = c
            observation_hat[:context_frames, :, i, :, [0, -1]] = c
            observation_gt[:, :, i, [0, -1], :] = c
            observation_gt[:, :, i, :, [0, -1]] = c
        video = torch.cat([observation_hat, observation_gt], -1).detach().cpu().numpy()
        video = np.transpose(np.clip(video, a_min=0.0, a_max=1.0) * 255, (1, 0, 2, 3, 4)).astype(np.uint8)
        # video[..., 1:] = video[..., :1]  # remove framestack, only visualize current frame
        n_samples = len(video)
        # use wandb directly here since pytorch lightning doesn't support logging videos yet
        for i in range(n_samples):
            logger.log(
                {
                    f"{namespace}/{prefix}_{i}": wandb.Video(video[i], fps=24),
                    f"trainer/global_step": step,
                }
            )
    elif observation_hat.size(2) == 8 and observation_hat.size(3) > 1:
        # load/default logger
        if not logger:
            logger = wandb
        if observation_gt is None:
            observation_gt = torch.zeros_like(observation_hat)
        # ensure context frames match GT
        observation_hat[:context_frames] = observation_gt[:context_frames]

        # convert to NumPy arrays of shape (B, T, C, H, W)
        vid_pred = observation_hat.detach().cpu().numpy().transpose(1, 0, 2, 3, 4)
        vid_gt   = observation_gt.detach().cpu().numpy().transpose(1, 0, 2, 3, 4)
        B, T, C, H, W = vid_pred.shape

        # global min/max for consistent scaling
        '''
        vmin = min(vid_pred.min(), vid_gt.min())
        vmax = max(vid_pred.max(), vid_gt.max())
        vid_pred = (vid_pred - vmin) / (vmax - vmin + 1e-8)
        vid_gt   = (vid_gt   - vmin) / (vmax - vmin + 1e-8)
        '''

        if namespace.startswith("train"):
            # training: log full (T × C) grid per sample
            for b in range(B):
                if b > 4: break
                fig, axs = plt.subplots(T, C, figsize=(C*2, T*2), squeeze=False)
                for t in range(T):
                    for ch in range(C):
                        ax = axs[t][ch]
                        vid_pred_norm_ch = (vid_pred[b,t,ch] - vid_pred[b,t,ch].min()) \
                                        / (vid_pred[b,t,ch].max() - vid_pred[b,t,ch].min() + 1e-8)
                        vid_gt_norm_ch = (vid_gt[b,t,ch] - vid_gt[b,t,ch].min()) \
                                        / (vid_gt[b,t,ch].max() - vid_gt[b,t,ch].min() + 1e-8)
                        pair = np.concatenate([vid_pred_norm_ch, vid_gt_norm_ch], axis=1)
                        # mn, mx = pair.min(), pair.max()
                        # pair = (pair - mn)/(mx - mn + 1e-8)
                        ax.imshow(pair, cmap="gray")
                        if t == 0:
                            ax.set_title(f"Ch {ch}")
                        ax.axis("off")
                plt.tight_layout()
                logger.log({
                    f"{namespace}/{prefix}_sample{b}": wandb.Image(fig),
                    "trainer/global_step": step,
                })
                plt.close(fig)

        else:
            # non-training: log one timestep at a time,
            # each as a 1×C row of subplots for channels
            for b in range(B):
                if b > 4: break  # limit to first 5 samples
                for t in range(12, 24):
                    fig, axs = plt.subplots(1, C, figsize=(C*2, 2), squeeze=False)
                    for ch in range(C):
                        ax = axs[0][ch]
                        vid_pred_norm_ch = (vid_pred[b,t,ch] - vid_pred[b,t,ch].min()) \
                                        / (vid_pred[b,t,ch].max() - vid_pred[b,t,ch].min() + 1e-8)
                        vid_gt_norm_ch = (vid_gt[b,t,ch] - vid_gt[b,t,ch].min()) \
                                        / (vid_gt[b,t,ch].max() - vid_gt[b,t,ch].min() + 1e-8)
                        pair = np.concatenate([vid_pred_norm_ch, vid_gt_norm_ch], axis=1)
                        # mn, mx = pair.min(), pair.max()
                        # pair = (pair - mn)/(mx - mn + 1e-8)
                        ax.imshow(pair, cmap="gray")
                        ax.set_title(f"Ch {ch}")
                        ax.axis("off")
                    plt.tight_layout()
                    logger.log({
                        f"{namespace}/{prefix}_sample{b}_frame{t}": wandb.Image(fig),
                        "trainer/global_step": step,
                    })
                    plt.close(fig)

    elif observation_hat.size(2) == 8 and observation_hat.size(3) == 1:
        
        # Convert the PyTorch tensors to NumPy arrays and squeeze out trailing singleton dimensions.
        # New shape: [batch_size, timesteps, 8]
        obs_hat = observation_hat.detach().cpu().numpy().squeeze(-1).squeeze(-1).transpose(1,0,2)
        obs_gt  = observation_gt.detach().cpu().numpy().squeeze(-1).squeeze(-1).transpose(1,0,2)
        
        batch_size, timesteps, _ = obs_hat.shape
        animations = []
        
        for b in range(batch_size):

            if b > 4: break

            # Extract positions for each object:
            # Object 1: indices 0 (x), 1 (y); Object 2: indices 4 (x), 5 (y).
            pos1_hat = obs_hat[b, :, 0:2]  # shape: (timesteps, 2)
            pos2_hat = obs_hat[b, :, 4:6]
            pos1_gt  = obs_gt[b, :, 0:2]
            pos2_gt  = obs_gt[b, :, 4:6]
            
            # Create a figure with two subplots (left for observation_hat, right for observation_gt)
            fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 5))
            margin = 0.05  # extra margin for axis limits
            
            # Configure left subplot axes (for observation_hat)
            positions_left = np.concatenate([pos1_hat, pos2_hat], axis=0)
            x_min_left, x_max_left = positions_left[:, 0].min(), positions_left[:, 0].max()
            y_min_left, y_max_left = positions_left[:, 1].min(), positions_left[:, 1].max()
            ax_left.set_xlim(x_min_left - margin, x_max_left + margin)
            ax_left.set_ylim(y_min_left - margin, y_max_left + margin)
            ax_left.set_title("Predictions")
            ax_left.set_xlabel("X")
            ax_left.set_ylabel("Y")
            
            # Configure right subplot axes (for observation_gt)
            positions_right = np.concatenate([pos1_gt, pos2_gt], axis=0)
            x_min_right, x_max_right = positions_right[:, 0].min(), positions_right[:, 0].max()
            y_min_right, y_max_right = positions_right[:, 1].min(), positions_right[:, 1].max()
            ax_right.set_xlim(x_min_right - margin, x_max_right + margin)
            ax_right.set_ylim(y_min_right - margin, y_max_right + margin)
            ax_right.set_title("Ground-Truth")
            ax_right.set_xlabel("X")
            ax_right.set_ylabel("Y")
            
            # Initialize line objects for each object in both subplots.
            line1_left, = ax_left.plot([], [], 'o-', color='blue', label='Object 1')
            line2_left, = ax_left.plot([], [], 'o-', color='orange', label='Object 2')
            ax_left.legend()
            
            line1_right, = ax_right.plot([], [], 'o-', color='blue', label='Object 1')
            line2_right, = ax_right.plot([], [], 'o-', color='orange', label='Object 2')
            ax_right.legend()
            
            # Create red border rectangles:
            # For left subplot, the red border should be present only during the first context_frames.
            rect_left = plt.Rectangle(
                (x_min_left - margin, y_min_left - margin),
                (x_max_left - x_min_left) + 2 * margin,
                (y_max_left - y_min_left) + 2 * margin,
                fill=False, edgecolor='red', linewidth=6
            )
            ax_left.add_patch(rect_left)
            
            # For right subplot, the red border is always visible.
            rect_right = plt.Rectangle(
                (x_min_right - margin, y_min_right - margin),
                (x_max_right - x_min_right) + 2 * margin,
                (y_max_right - y_min_right) + 2 * margin,
                fill=False, edgecolor='red', linewidth=6
            )
            ax_right.add_patch(rect_right)
            
            # Initialization function for the animation.
            def init():
                line1_left.set_data([], [])
                line2_left.set_data([], [])
                line1_right.set_data([], [])
                line2_right.set_data([], [])
                rect_left.set_visible(True)  # initially visible for left
                rect_right.set_visible(True)  # always visible for right
                return line1_left, line2_left, line1_right, line2_right, rect_left, rect_right
            
            # Update function: updates the traces and toggles visibility of the left red border.
            def update(frame):
                line1_left.set_data(pos1_hat[:frame+1, 0], pos1_hat[:frame+1, 1])
                line2_left.set_data(pos2_hat[:frame+1, 0], pos2_hat[:frame+1, 1])
                line1_right.set_data(pos1_gt[:frame+1, 0], pos1_gt[:frame+1, 1])
                line2_right.set_data(pos2_gt[:frame+1, 0], pos2_gt[:frame+1, 1])
                
                # Left subplot: red border is visible only for frames less than context_frames.
                rect_left.set_visible(frame < context_frames)
                # Right subplot: red border remains visible.
                rect_right.set_visible(True)
                
                return line1_left, line2_left, line1_right, line2_right, rect_left, rect_right
            
            # Create the animation object.
            fps=12
            ani = FuncAnimation(
                fig,
                update,
                frames=timesteps,
                init_func=init,
                blit=True,
                interval=1000 / fps  # milliseconds per frame
            )
            
            # Optionally, save the animation to a video file. If logger is provided, log the video to wandb.
            video_path = None
            output_prefix = None
            if output_prefix is not None:
                video_path = f"{output_prefix}_batch_{b}.mp4"
            elif logger is not None:
                # If no output_prefix is provided and logger is available, use a temporary file.
                tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                video_path = tmp_file.name
                tmp_file.close()
            
            if video_path is not None:
                ani.save(video_path, writer='ffmpeg', fps=fps)
                print(f"Saved video to {video_path}")
                if logger is not None:
                    # Log the video to wandb.
                    logger.log({f"{namespace}/{prefix}_{b}": wandb.Video(video_path, fps=fps, format="mp4"), 
                                f"trainer/global_step": step,})
                    # If we used a temporary file (i.e. no output_prefix), remove it after logging.
                    if output_prefix is None:
                        os.remove(video_path)
            
            animations.append(ani)
            plt.close(fig)


def get_validation_metrics_for_videos(
    observation_hat,
    observation_gt,
    metrics=["mse", "psnr", "ssim", "uiqi"],
    lpips_model: Optional[LearnedPerceptualImagePatchSimilarity] = None,
    fid_model: Optional[FrechetInceptionDistance] = None,
    fvd_model: Optional[FrechetVideoDistance] = None,
):
    """
    :param observation_hat: predicted observation tensor of shape (frame, batch, channel, height, width)
    :param observation_gt: ground-truth observation tensor of shape (frame, batch, channel, height, width)
    :param lpips_model: a LearnedPerceptualImagePatchSimilarity object from algorithm.common.metrics
    :param fid_model: a FrechetInceptionDistance object  from algorithm.common.metrics
    :param fvd_model: a FrechetVideoDistance object  from algorithm.common.metrics
    :return: a tuple of metrics
    """
    frame, batch, channel, height, width = observation_hat.shape
    output_dict = {}
    observation_gt = observation_gt.type_as(observation_hat)  # some metrics don't fully support fp16

    if frame < 9:
        fvd_model = None  # FVD requires at least 9 frames

    if fvd_model is not None:
        output_dict["fvd"] = fvd_model.compute(
            torch.clamp(observation_hat, -1.0, 1.0),
            torch.clamp(observation_gt, -1.0, 1.0),
        )

    # reshape to (frame * batch, channel, height, width) for image losses
    observation_hat = observation_hat.view(-1, channel, height, width)
    observation_gt = observation_gt.view(-1, channel, height, width)

    if "mse" in metrics:
        output_dict["mse"] = mean_squared_error(observation_hat, observation_gt)
    if "psnr" in metrics:
        output_dict["psnr"] = peak_signal_noise_ratio(observation_hat, observation_gt, data_range=2.0)
    if "ssim" in metrics:
        output_dict["ssim"] = structural_similarity_index_measure(observation_hat, observation_gt, data_range=2.0)
    if "uiqi" in metrics:
        output_dict["uiqi"] = universal_image_quality_index(observation_hat, observation_gt)
    
    # operations for LPIPS and FID
    observation_hat = torch.clamp(observation_hat, -1.0, 1.0)
    observation_gt = torch.clamp(observation_gt, -1.0, 1.0)

    if lpips_model is not None:
        lpips_model.update(observation_hat, observation_gt)
        lpips = lpips_model.compute().item()
        # Reset the states of non-functional metrics
        output_dict["lpips"] = lpips
        lpips_model.reset()

    if fid_model is not None:
        observation_hat_uint8 = ((observation_hat + 1.0) / 2 * 255).type(torch.uint8)
        observation_gt_uint8 = ((observation_gt + 1.0) / 2 * 255).type(torch.uint8)
        fid_model.update(observation_gt_uint8, real=True)
        fid_model.update(observation_hat_uint8, real=False)
        fid = fid_model.compute()
        output_dict["fid"] = fid
        # Reset the states of non-functional metrics
        fid_model.reset()

    return output_dict


def is_grid_env(env_id):
    return "maze2d" in env_id or "diagonal2d" in env_id


def get_maze_grid(env_id):
    # import gym
    # maze_string = gym.make(env_id).str_maze_spec
    if "large" in env_id:
        maze_string = "############\\#OOOO#OOOOO#\\#O##O#O#O#O#\\#OOOOOO#OOO#\\#O####O###O#\\#OO#O#OOOOO#\\##O#O#O#O###\\#OO#OOO#OGO#\\############"
    if "medium" in env_id:
        maze_string = "########\\#OO##OO#\\#OO#OOO#\\##OOO###\\#OO#OOO#\\#O#OO#O#\\#OOO#OG#\\########"
    if "umaze" in env_id:
        maze_string = "#####\\#GOO#\\###O#\\#OOO#\\#####"
    lines = maze_string.split("\\")
    grid = [line[1:-1] for line in lines]
    return grid[1:-1]


def get_random_start_goal(env_id, batch_size):
    maze_grid = get_maze_grid(env_id)
    s2i = {"O": 0, "#": 1, "G": 2}
    maze_grid = [[s2i[s] for s in r] for r in maze_grid]
    maze_grid = np.array(maze_grid)
    x, y = np.nonzero(maze_grid == 0)
    indices = np.random.randint(len(x), size=batch_size)
    start = np.stack([x[indices], y[indices]], -1) + 1
    x, y = np.nonzero(maze_grid == 2)
    goal = np.concatenate([x, y], -1)
    goal = np.tile(goal[None, :], (batch_size, 1)) + 1
    return start, goal


def plot_maze_layout(ax, maze_grid):
    ax.clear()

    if maze_grid is not None:
        for i, row in enumerate(maze_grid):
            for j, cell in enumerate(row):
                if cell == "#":
                    square = plt.Rectangle((i + 0.5, j + 0.5), 1, 1, edgecolor="black", facecolor="black")
                    ax.add_patch(square)

    ax.set_aspect("equal")
    ax.grid(True, color="white", linewidth=4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_linewidth(4)
    ax.spines["right"].set_linewidth(4)
    ax.spines["bottom"].set_linewidth(4)
    ax.spines["left"].set_linewidth(4)
    ax.set_facecolor("lightgray")
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.set_xticks(np.arange(0.5, len(maze_grid) + 0.5))
    ax.set_yticks(np.arange(0.5, len(maze_grid[0]) + 0.5))
    ax.set_xlim(0.5, len(maze_grid) + 0.5)
    ax.set_ylim(0.5, len(maze_grid[0]) + 0.5)
    ax.grid(True, color="white", which="minor", linewidth=4)


def plot_start_goal(ax, start_goal: None):
    def draw_star(center, radius, num_points=5, color="black"):
        angles = np.linspace(0.0, 2 * np.pi, num_points, endpoint=False) + 5 * np.pi / (2 * num_points)
        inner_radius = radius / 2.0

        points = []
        for angle in angles:
            points.extend(
                [
                    center[0] + radius * np.cos(angle),
                    center[1] + radius * np.sin(angle),
                    center[0] + inner_radius * np.cos(angle + np.pi / num_points),
                    center[1] + inner_radius * np.sin(angle + np.pi / num_points),
                ]
            )

        star = plt.Polygon(np.array(points).reshape(-1, 2), color=color)
        ax.add_patch(star)

    start_x, start_y = start_goal[0]
    start_outer_circle = plt.Circle((start_x, start_y), 0.16, facecolor="white", edgecolor="black")
    ax.add_patch(start_outer_circle)
    start_inner_circle = plt.Circle((start_x, start_y), 0.08, color="black")
    ax.add_patch(start_inner_circle)

    goal_x, goal_y = start_goal[1]
    goal_outer_circle = plt.Circle((goal_x, goal_y), 0.16, facecolor="white", edgecolor="black")
    ax.add_patch(goal_outer_circle)
    draw_star((goal_x, goal_y), radius=0.08)


def make_trajectory_images(env_id, trajectory, batch_size, start, goal, plot_end_points=True):
    images = []
    for batch_idx in range(batch_size):
        fig, ax = plt.subplots()
        if is_grid_env(env_id):
            maze_grid = get_maze_grid(env_id)
        else:
            maze_grid = None
        plot_maze_layout(ax, maze_grid)
        ax.scatter(trajectory[:, batch_idx, 0], trajectory[:, batch_idx, 1], c=np.arange(len(trajectory)), cmap="Reds"),
        if plot_end_points:
            start_goal = (start[batch_idx], goal[batch_idx])
            plot_start_goal(ax, start_goal)
        # plt.title(f"sample_{batch_idx}")
        fig.tight_layout()
        fig.canvas.draw()
        img_shape = fig.canvas.get_width_height()[::-1] + (4,)
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).copy().reshape(img_shape)
        images.append(img)

        plt.close()
    return images


def make_convergence_animation(
    env_id,
    plan_history,
    trajectory,
    start,
    goal,
    open_loop_horizon,
    namespace,
    interval=100,
    plot_end_points=True,
    batch_idx=0,
):
    # - plan_history: contains for each time step all the MPC predicted plans for each pyramid noise level.
    #                 Structured as a list of length (episode_len // open_loop_horizon), where each
    #                 element corresponds to a control_time_step and stores a list of length pyramid_height,
    #                 where each element is a plan at a different pyramid noise level and stored as a tensor of
    #                 shape (episode_len // open_loop_horizon - control_time_step,
    #                        batch_size, x_stacked_shape)

    # select index and prune history
    start, goal = start[batch_idx], goal[batch_idx]
    trajectory = trajectory[:, batch_idx]
    plan_history = [[pm[:, batch_idx] for pm in pt] for pt in plan_history]
    trajectory, plan_history = prune_history(plan_history, trajectory, goal, open_loop_horizon)

    # animate the convergence of the first plan
    fig, ax = plt.subplots()
    if "large" in env_id:
        fig.set_size_inches(3.5, 5)
    else:
        fig.set_size_inches(3, 3)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

    if is_grid_env(env_id):
        maze_grid = get_maze_grid(env_id)
    else:
        maze_grid = None

    def update(frame):
        plot_maze_layout(ax, maze_grid)

        plan_history_m = plan_history[0][frame]
        plan_history_m = plan_history_m.numpy()
        ax.scatter(
            plan_history_m[:, 0],
            plan_history_m[:, 1],
            c=np.arange(len(plan_history_m))[::-1],
            cmap="Reds",
        )

        if plot_end_points:
            plot_start_goal(ax, (start, goal))

    frames = tqdm(range(len(plan_history[0])), desc="Making convergence animation")
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval)
    prefix = wandb.run.id if wandb.run is not None else env_id
    filename = f"/tmp/{prefix}_{namespace}_convergence.mp4"
    ani.save(filename, writer="ffmpeg", fps=24)
    return filename


def prune_history(plan_history, trajectory, goal, open_loop_horizon):
    dist = np.linalg.norm(
        trajectory[:, :2] - np.array(goal)[None],
        axis=-1,
    )
    reached = dist < 0.2
    if reached.any():
        cap_idx = np.argmax(reached)
        trajectory = trajectory[: cap_idx + open_loop_horizon + 1]
        plan_history = plan_history[: cap_idx // open_loop_horizon + 2]

    pruned_plan_history = []
    for plans in plan_history:
        pruned_plan_history.append([])
        for m in range(len(plans)):
            plan = plans[m]
            pruned_plan_history[-1].append(plan)
        plan = pruned_plan_history[-1][-1]
        dist = np.linalg.norm(plan.numpy()[:, :2] - np.array(goal)[None], axis=-1)
        reached = dist < 0.2
        if reached.any():
            cap_idx = np.argmax(reached) + 1
            pruned_plan_history[-1] = [p[:cap_idx] for p in pruned_plan_history[-1]]
    return trajectory, pruned_plan_history


def make_mpc_animation(
    env_id,
    plan_history,
    trajectory,
    start,
    goal,
    open_loop_horizon,
    namespace,
    interval=100,
    plot_end_points=True,
    batch_idx=0,
):
    # - plan_history: contains for each time step all the MPC predicted plans for each pyramid noise level.
    #                 Structured as a list of length (episode_len // open_loop_horizon), where each
    #                 element corresponds to a control_time_step and stores a list of length pyramid_height,
    #                 where each element is a plan at a different pyramid noise level and stored as a tensor of
    #                 shape (episode_len // open_loop_horizon - control_time_step,
    #                        batch_size, x_stacked_shape)

    # select index and prune history
    start, goal = start[batch_idx], goal[batch_idx]
    trajectory = trajectory[:, batch_idx]
    plan_history = [[pm[:, batch_idx] for pm in pt] for pt in plan_history]
    trajectory, plan_history = prune_history(plan_history, trajectory, goal, open_loop_horizon)

    # animate the convergence of the plans
    fig, ax = plt.subplots()
    if "large" in env_id:
        fig.set_size_inches(3.5, 5)
    else:
        fig.set_size_inches(3, 3)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    trajectory_colors = np.linspace(0, 1, len(trajectory))

    if is_grid_env(env_id):
        maze_grid = get_maze_grid(env_id)
    else:
        maze_grid = None

    def update(frame):
        control_time_step = 0
        while frame >= 0:
            frame -= len(plan_history[control_time_step])
            control_time_step += 1
        control_time_step -= 1
        m = frame + len(plan_history[control_time_step])
        num_steps_taken = 1 + open_loop_horizon * control_time_step
        plot_maze_layout(ax, maze_grid)

        plan_history_m = plan_history[control_time_step][m]
        plan_history_m = plan_history_m.numpy()
        ax.scatter(
            trajectory[:num_steps_taken, 0],
            trajectory[:num_steps_taken, 1],
            c=trajectory_colors[:num_steps_taken],
            cmap="Blues",
        )
        ax.scatter(
            plan_history_m[:, 0],
            plan_history_m[:, 1],
            c=np.arange(len(plan_history_m))[::-1],
            cmap="Reds",
        )

        if plot_end_points:
            plot_start_goal(ax, (start, goal))

    num_frames = sum([len(p) for p in plan_history])
    frames = tqdm(range(num_frames), desc="Making MPC animation")
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval)
    prefix = wandb.run.id if wandb.run is not None else env_id
    filename = f"/tmp/{prefix}_{namespace}_mpc.mp4"
    ani.save(filename, writer="ffmpeg", fps=24)

    return filename
