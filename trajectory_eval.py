import numpy as np
import matplotlib.pyplot as plt

def calculate_velocity(preds, ground_truth, group):
    """
    Calculates the velocity of two objects given their x, y positions for both predictions and ground-truth.
    Concatenates the velocities for both objects, averages over 'group' consecutive timesteps, and plots
    the difference in momentum for both prediction and ground-truth velocities.

    Parameters:
    preds (np.array): Predictions array of shape [batch_size, timesteps, 8].
    ground_truth (np.array): Ground-truth array of shape [batch_size, timesteps, 8].
    group (int): The number of consecutive timesteps to average over.

    Returns:
    vel_pred (np.array): Concatenated and averaged velocity of both objects from predictions, shape [batch_size, (timesteps - 1) / group, 4].
    vel_gt (np.array): Concatenated and averaged velocity of both objects from ground truth, shape [batch_size, (timesteps - 1) / group, 4].
    """
    
    # Extract the x and y position columns for object 1 and object 2 in both predictions and ground-truth
    preds_obj1_pos = preds[:, :, 0:2]  # Extracting x,y positions for object 1 (predictions)
    preds_obj2_pos = preds[:, :, 4:6]  # Extracting x,y positions for object 2 (predictions)
    gt_obj1_pos = ground_truth[:, :, 0:2]  # Extracting x,y positions for object 1 (ground-truth)
    gt_obj2_pos = ground_truth[:, :, 4:6]  # Extracting x,y positions for object 2 (ground-truth)
    
    # Calculate velocity by subtracting consecutive positions (position difference) for predictions
    vel_obj1_pred = preds_obj1_pos[:, 1:, :] - preds_obj1_pos[:, :-1, :]
    vel_obj2_pred = preds_obj2_pos[:, 1:, :] - preds_obj2_pos[:, :-1, :]
    
    # Calculate velocity by subtracting consecutive positions (position difference) for ground-truth
    vel_obj1_gt = gt_obj1_pos[:, 1:, :] - gt_obj1_pos[:, :-1, :]
    vel_obj2_gt = gt_obj2_pos[:, 1:, :] - gt_obj2_pos[:, :-1, :]
    
    # Concatenate velocities for both objects (x, y velocities)
    vel_pred = np.concatenate([vel_obj1_pred, vel_obj2_pred], axis=-1)  # Shape: [batch_size, timesteps - 1, 4]
    vel_gt = np.concatenate([vel_obj1_gt, vel_obj2_gt], axis=-1)  # Shape: [batch_size, timesteps - 1, 4]
    
    # Average over 'group' consecutive timesteps
    if group > 1:
        # Calculate the number of new timesteps after grouping
        new_timesteps = vel_pred.shape[1] // group
        
        # Reshape and average over the group dimension
        vel_pred = vel_pred.reshape(vel_pred.shape[0], new_timesteps, group, 4).mean(axis=2)
        vel_gt = vel_gt.reshape(vel_gt.shape[0], new_timesteps, group, 4).mean(axis=2)
    
    # Calculate difference in momentum between consecutive timesteps for both predictions and ground-truth
    momentum_pred_diff = np.sum(np.abs(vel_pred[:, 1:, :] - vel_pred[:, :-1, :]), axis=(1, 2))
    momentum_gt_diff = np.sum(np.abs(vel_gt[:, 1:, :] - vel_gt[:, :-1, :]), axis=(1, 2))

    # Plot the difference in momentum for both predictions and ground-truth
    timesteps = np.arange(1, momentum_pred_diff.shape[0] + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, momentum_pred_diff, label='Momentum Difference (Pred)', color='b')
    plt.plot(timesteps, momentum_gt_diff, label='Momentum Difference (GT)', color='r')
    
    plt.xlabel('Timestep')
    plt.ylabel('Momentum Difference')
    plt.title('Difference in Momentum (Predictions vs Ground Truth)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return vel_pred, vel_gt
