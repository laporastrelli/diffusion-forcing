import numpy as np
import matplotlib.pyplot as plt
import os

import sys

##### MOMENTUM #####
def check_collision(ball1_position, ball2_position, box_details, 
                    pre_collision_velocities, post_collision_velocities, radii):
    """
    Verify whether one of the three following events has occurred:
    1. Ball1 has collided with the wall
    2. Ball2 has collided with the wall
    3. Ball1 and Ball2 have collided with each other
    4. Ball1 and Ball2 have not collided

    :param ball1_position: Position of ball1 (x, y)
    :param ball2_position: Position of ball2 (x, y)
    :param box_details: Details of the box (width, height, line_width)
    :param pre_collision_velocities: Pre-collision velocities of both balls [(vx1, vy1), (vx2, vy2)]
    :param post_collision_velocities: Post-collision velocities of both balls [(vx1, vy1), (vx2, vy2)]
    :return: Dictionary with the type of collision as key and an array containing the pre and post collision velocities
    """
    
    box_width, box_height, line_width = box_details
    r1, r2 = radii
    collision_data = {}

    collision_occurred = False
    # Check for collision with walls for ball1
    if ball1_position[0] <= line_width + r1 or ball1_position[0] >= box_width - line_width - r1 or \
       ball1_position[1] <= line_width + r1 or ball1_position[1] >= box_height - line_width - r1:
        collision_data['wall_collision_ball_1'] = {
            'pre_collision_velocity': pre_collision_velocities[0],
            'post_collision_velocity': post_collision_velocities[0]
        }
        collision_occurred = True

    # Check for collision with walls for ball2
    if ball2_position[0] <= line_width + r2 or ball2_position[0] >= box_width - line_width - r2 or \
       ball2_position[1] <= line_width + r2 or ball2_position[1] >= box_height - line_width - r2 :
        collision_data['wall_collision_ball_2'] = {
            'pre_collision_velocity': pre_collision_velocities[1],
            'post_collision_velocity': post_collision_velocities[1]
        }
        collision_occurred = True

    # Check for collision between balls
    distance_between_balls = np.linalg.norm(np.array(ball1_position) - np.array(ball2_position))
    # print(distance_between_balls)
    if distance_between_balls <= (r1+r2):
        collision_data['ball_collision'] = {
            'pre_collision_velocity_ball_1': pre_collision_velocities[0],
            'post_collision_velocity_ball_1': post_collision_velocities[0],
            'pre_collision_velocity_ball_2': pre_collision_velocities[1],
            'post_collision_velocity_ball_2': post_collision_velocities[1]
        }
        collision_occurred = True
    
    # store velocities for both balls 
    if not collision_occurred:
        collision_data['no_collision'] = {
            'pre_collision_velocity_ball_1': pre_collision_velocities[0],
            'post_collision_velocity_ball_1': post_collision_velocities[0],
            'pre_collision_velocity_ball_2': pre_collision_velocities[1],
            'post_collision_velocity_ball_2': post_collision_velocities[1]
        }

    return collision_data

def plot_momentum(all_simulations_collision_data, plot_info):
    dir_out, context_length, name_out = plot_info

    # Initialize lists to store momentum data for each collision type
    ball_collision_momentum_x = []
    ball_collision_momentum_y = []

    wall_collision_ball_1_momentum_x = []
    wall_collision_ball_1_momentum_y = []

    wall_collision_ball_2_momentum_x = []
    wall_collision_ball_2_momentum_y = []

    no_coll_ball_1_i = []
    no_coll_ball_1_f = []
    no_coll_ball_2_i = []
    no_coll_ball_2_f = []

    # Iterate over each simulation
    for sim_key, collision_data in all_simulations_collision_data.items():

        # Extract radii from the key
        if sim_key.find('r1')!= -1 or sim_key.find('r2')!= -1:
            key_parts = sim_key.split('_')
            r1 = float(key_parts[3])
            r2 = float(key_parts[5])
        else:
            r1 = r2 = 1

        # Iterate over each step in the collision data
        for step_key, collision_info in collision_data.items():
            if 'ball_collision' in collision_info:
                pre_collision_velocity_ball_1 = collision_info['ball_collision']['pre_collision_velocity_ball_1']
                post_collision_velocity_ball_1 = collision_info['ball_collision']['post_collision_velocity_ball_1']
                pre_collision_velocity_ball_2 = collision_info['ball_collision']['pre_collision_velocity_ball_2']
                post_collision_velocity_ball_2 = collision_info['ball_collision']['post_collision_velocity_ball_2']

                # Calculate initial and final momentum components for both balls
                p1_i_x, p1_i_y = r1 * pre_collision_velocity_ball_1[0], r1 * pre_collision_velocity_ball_1[1]
                p2_i_x, p2_i_y = r2 * pre_collision_velocity_ball_2[0], r2 * pre_collision_velocity_ball_2[1]
                p1_f_x, p1_f_y = r1 * post_collision_velocity_ball_1[0], r1 * post_collision_velocity_ball_1[1]
                p2_f_x, p2_f_y = r2 * post_collision_velocity_ball_2[0], r2 * post_collision_velocity_ball_2[1]

                ball_collision_momentum_x.append(p1_i_x + p2_i_x)
                ball_collision_momentum_y.append(p1_f_x + p2_f_x)
                ball_collision_momentum_x.append(p1_i_y + p2_i_y)
                ball_collision_momentum_y.append(p1_f_y + p2_f_y)

            elif 'wall_collision_ball_1' in collision_info:
                pre_collision_velocity = collision_info['wall_collision_ball_1']['pre_collision_velocity']
                post_collision_velocity = collision_info['wall_collision_ball_1']['post_collision_velocity']

                # Calculate initial and final momentum components for ball 1
                p_i_x, p_i_y = r1 * pre_collision_velocity[0], r1 * pre_collision_velocity[1]
                p_f_x, p_f_y = r1 * post_collision_velocity[0], r1 * post_collision_velocity[1]

                wall_collision_ball_1_momentum_x.append(p_i_x)
                wall_collision_ball_1_momentum_y.append(p_f_x)
                wall_collision_ball_1_momentum_x.append(p_i_y)
                wall_collision_ball_1_momentum_y.append(p_f_y)

            elif 'wall_collision_ball_2' in collision_info:
                pre_collision_velocity = collision_info['wall_collision_ball_2']['pre_collision_velocity']
                post_collision_velocity = collision_info['wall_collision_ball_2']['post_collision_velocity']

                # Calculate initial and final momentum components for ball 2
                p_i_x, p_i_y = r2 * pre_collision_velocity[0], r2 * pre_collision_velocity[1]
                p_f_x, p_f_y = r2 * post_collision_velocity[0], r2 * post_collision_velocity[1]
               
                wall_collision_ball_2_momentum_x.append(p_i_x)
                wall_collision_ball_2_momentum_y.append(p_f_x)
                wall_collision_ball_2_momentum_x.append(p_i_y)
                wall_collision_ball_2_momentum_y.append(p_f_y)
            
            else:
                pre_collision_velocity_ball_1 = collision_info['no_collision']['pre_collision_velocity_ball_1']
                pre_collision_velocity_ball_2 = collision_info['no_collision']['pre_collision_velocity_ball_2']
                post_collision_velocity_ball_1 = collision_info['no_collision']['post_collision_velocity_ball_1']
                post_collision_velocity_ball_2 = collision_info['no_collision']['post_collision_velocity_ball_2']

                p1_i = pre_collision_velocity_ball_1[0] + pre_collision_velocity_ball_1[1]
                p1_f = post_collision_velocity_ball_1[0] + post_collision_velocity_ball_1[1]
                p2_i = pre_collision_velocity_ball_2[0] + pre_collision_velocity_ball_2[1]
                p2_f = post_collision_velocity_ball_2[0] + post_collision_velocity_ball_2[1]

                no_coll_ball_1_i.append(p1_i)
                no_coll_ball_1_f.append(p1_f)
                no_coll_ball_2_i.append(p2_i)
                no_coll_ball_2_f.append(p2_f)
  
    # Plot for two balls colliding with each other
    fig1 = plt.figure()
    plt.scatter(ball_collision_momentum_x[:len(ball_collision_momentum_x)//2], 
                ball_collision_momentum_y[:len(ball_collision_momentum_y)//2], 
                label='x momentum', color='blue')
    plt.scatter(ball_collision_momentum_x[len(ball_collision_momentum_x)//2:], 
                ball_collision_momentum_y[len(ball_collision_momentum_y)//2:], 
                label='y momentum', color='red')
    plt.xlabel('Initial Momentum')
    plt.ylabel('Final Momentum')
    plt.legend()
    plt.title('Two Balls Colliding with Each Other (Context={})'.format(context_length))
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    fig1.savefig(os.path.join(dir_out, name_out + '_two_ball_collision.jpg'))

    # Plot for ball 1 colliding with the wall
    fig2 = plt.figure()
    plt.scatter(wall_collision_ball_1_momentum_x[:len(wall_collision_ball_1_momentum_x)//2], 
                wall_collision_ball_1_momentum_y[:len(wall_collision_ball_1_momentum_y)//2], 
                label='x momentum', color='blue')
    plt.scatter(wall_collision_ball_1_momentum_x[len(wall_collision_ball_1_momentum_x)//2:], 
                wall_collision_ball_1_momentum_y[len(wall_collision_ball_1_momentum_y)//2:], 
                label='y momentum', color='red')
    plt.xlabel('Initial Momentum')
    plt.ylabel('Final Momentum')
    plt.legend()
    plt.title('Ball 1 Colliding with the Wall (Context={})'.format(context_length))
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    fig2.savefig(os.path.join(dir_out, name_out + '_ball1_collision.jpg'))

    # Plot for ball 2 colliding with the wall
    fig3 = plt.figure()
    plt.scatter(wall_collision_ball_2_momentum_x[::2], 
                wall_collision_ball_2_momentum_y[::2], 
                label='x momentum', color='blue')
    plt.scatter(wall_collision_ball_2_momentum_x[1::2], 
                wall_collision_ball_2_momentum_y[1::2], 
                label='y momentum', color='red')
    plt.xlabel('Initial Momentum')
    plt.ylabel('Final Momentum')
    plt.legend()
    plt.title('Ball 2 Colliding with the Wall (Context={})'.format(context_length))
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    fig3.savefig(os.path.join(dir_out, name_out + '_ball2_collision.jpg'))

    fig4 = plt.figure()
    plt.scatter(no_coll_ball_1_i, no_coll_ball_1_f, 
                label='ball 1', color='blue')
    plt.scatter(no_coll_ball_2_i, no_coll_ball_2_f, 
                label='ball 2', color='red')
    plt.xlabel('Initial Momentum')
    plt.ylabel('Final Momentum')
    plt.legend()
    plt.title('No Collision (Context={})'.format(context_length))
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    fig3.savefig(os.path.join(dir_out, name_out + '_no_collision.jpg'))

def store_collisions(arr):
    box_details = (64, 64, 1)
    radii = [6,6]
    all_simulations_collision_data = {}

    for b in range(arr.shape[0]):

        collision_data_all_steps = {}
        for t in range(1, arr.shape[1]):
            ball1_position = arr[b, t, 0:2]
            ball2_position = arr[b, t, 4:6]
            pre_collision_velocities = [(arr[b, t-1, 2], arr[b, t-1, 3]), \
                                        (arr[b, t-1, 6], arr[b, t-1, 7])]
            post_collision_velocities = [(arr[b, t, 2], arr[b, t, 3]), \
                                        (arr[b, t, 6], arr[b, t, 7])]
            collision_data_step_i = check_collision(ball1_position, ball2_position, box_details, 
                    pre_collision_velocities, post_collision_velocities, radii)
            collision_data_all_steps[f'step_{t}'] = collision_data_step_i
        all_simulations_collision_data[f'simulation_{b}'] = collision_data_all_steps

    return all_simulations_collision_data

def show_momentum(preds, ground_truth, save_dir, context_length):

    momentum_data_preds = store_collisions(preds)
    momentum_data_gt = store_collisions(ground_truth)
    
    plot_momentum(momentum_data_preds, plot_info = [save_dir, context_length, 'preds'])
    plot_momentum(momentum_data_gt, plot_info = [save_dir, context_length, 'gts'])

##### MSE #####
def calculate_mse_and_plot(preds, ground_truth, save_dir, context_length):
    """
    Calculates the mean squared error (MSE) between preds and ground_truth across time and batch size,
    then averages it over the contiguous pairs of the 8 components (x, y position and velocity for both objects),
    resulting in a [timesteps, 4] array. The function then plots two side-by-side plots showing the MSE over time
    for positions and velocities.

    Parameters:
    preds (np.array): Predictions array of shape [batch_size, timesteps, 8].
    ground_truth (np.array): Ground-truth array of shape [batch_size, timesteps, 8].

    Returns:
    mse_avg (np.array): Averaged MSE for position and velocity, shape [timesteps, 4].
    """
    
    # Calculate the MSE between predictions and ground-truth for all 8 components
    mse = np.mean((preds - ground_truth) ** 2, axis=0)  # shape [timesteps, 8]

    # A more explicit way to average across contiguous pairs of columns (x, y for position and velocity)
    mse_avg = np.zeros((mse.shape[0], 4))  # Initialize an array of shape [timesteps, 4]

    # Average position MSE for object 1 (x and y)
    mse_avg[:, 0] = np.mean(mse[:, 0:2], axis=1)
    # Average velocity MSE for object 1 (x and y)
    mse_avg[:, 1] = np.mean(mse[:, 2:4], axis=1)
    # Average position MSE for object 2 (x and y)
    mse_avg[:, 2] = np.mean(mse[:, 4:6], axis=1)
    # Average velocity MSE for object 2 (x and y)
    mse_avg[:, 3] = np.mean(mse[:, 6:8], axis=1)

    # Plotting the MSE for positions and velocities
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plotting the MSE for positions (x, y) for both objects
    # Plotting the MSE for positions (x, y) for both objects
    ax1.plot(np.arange(mse_avg.shape[0]), mse_avg[:, 0], label="Obj 1 Position", color='b')
    ax1.plot(np.arange(mse_avg.shape[0]), mse_avg[:, 2], label="Obj 2 Position", color='r')
    ax1.set_title("Mean Squared Error (Position) Over Time")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Mean Squared Error")
    ax1.legend()
    ax1.grid(True)

    # Plotting the MSE for velocities (x, y) for both objects
    ax2.plot(np.arange(mse_avg.shape[0]), mse_avg[:, 1], label="Obj 1 Velocity", color='g')
    ax2.plot(np.arange(mse_avg.shape[0]), mse_avg[:, 3], label="Obj 2 Velocity", color='y')
    ax2.set_title("Mean Squared Error (Velocity) Over Time - Context={}".format(context_length))
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Mean Squared Error")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    fig.savefig(os.path.join(save_dir, 'mse_time.jpg'))

    return mse_avg

if __name__ == "__main__":

    root_path = "/data2/users/lr4617/video_generation/diffusion-forcing/outputs/2025-02-26/16-15-48/validation"
    path_to_preds = os.path.join(root_path, "validation_preds.npy")
    path_to_gts = os.path.join(root_path, "validation_gts.npy")

    # context_length = int(root_path.split('/')[-1].split('_')[-1])
    context_length = 48

    preds = np.load(path_to_preds)
    gts = np.load(path_to_gts)
    preds = np.transpose(np.squeeze(np.squeeze(preds, axis=4), axis=3), (1, 0, 2))
    gts = np.transpose(np.squeeze(np.squeeze(gts, axis=4), axis=3), (1, 0, 2))

    print(gts.shape)
    print(gts[0, 0],
          gts[0, 1], 
          gts[0, 2], 
          gts[0, 3])
    sys.exit()

    show_momentum(preds=preds, 
                  ground_truth=gts, 
                  save_dir=root_path, 
                  context_length=context_length)
    
    calculate_mse_and_plot(preds=preds, 
                           ground_truth=gts, 
                           save_dir=root_path, 
                           context_length=context_length)
    


'''
def calculate_momentum(preds, ground_truth, save_dir, context_length):
    # Extract the velocity columns for object 1 and object 2 in both predictions and ground-truth
    preds_obj1_vel = preds[:, :, 2:4]  # Extracting x,y velocities for object 1 (predictions)
    preds_obj2_vel = preds[:, :, 6:8]  # Extracting x,y velocities for object 2 (predictions)
    gt_obj1_vel = ground_truth[:, :, 2:4]  # Extracting x,y velocities for object 1 (ground-truth)
    gt_obj2_vel = ground_truth[:, :, 6:8]  # Extracting x,y velocities for object 2 (ground-truth)

    p_x_preds = preds_obj1_vel[:, :, 0] + preds_obj2_vel[:, :, 0]
    p_y_preds = preds_obj1_vel[:, :, 1] + preds_obj2_vel[:, :, 1]

    p_x_gt = gt_obj1_vel[:, :, 0] + gt_obj2_vel[:, :, 0]
    p_y_gt = gt_obj1_vel[:, :, 1] + gt_obj2_vel[:, :, 1]

    p_x_preds_arr = (np.stack([p_x_preds[:, :-1], p_x_preds[:, 1:]], axis=-1)).reshape(-1, 2)
    p_y_preds_arr = (np.stack([p_y_preds[:, :-1] , p_y_preds[:, 1:]], axis=-1)).reshape(-1, 2)

    p_x_gt_arr = (np.stack([p_x_gt[:, :-1], p_x_gt[:, 1:]], axis=-1)).reshape(-1, 2)
    p_y_gt_arr = (np.stack([p_y_gt[:, :-1], p_y_gt[:, 1:]], axis=-1)).reshape(-1, 2)


    p_x_preds_diff = (np.abs(p_x_preds[:, 1:] - p_x_preds[:, :-1])).reshape(-1,)
    p_y_preds_diff = (np.abs(p_y_preds[:, 1:] - p_y_preds[:, :-1])).reshape(-1,)

    p_x_gt_diff = (np.abs(p_x_gt[:, 1:] - p_x_gt[:, :-1])).reshape(-1,)
    p_y_gt_diff = (np.abs(p_y_gt[:, 1:] - p_y_gt[:, :-1])).reshape(-1,)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.scatter(p_x_preds_arr[:, 0], p_x_preds_arr[:, 1], label="x-momentum", color='b')
    ax1.scatter(p_y_preds_arr[:, 0], p_y_preds_arr[:, 1], label="y-momentum", color='r')
    ax1.set_title("Momemtum Relation - Predictions")
    ax1.set_xlabel("Initial Momemtum")
    ax1.set_ylabel("Final Momentum")
    ax1.legend()
    ax1.grid(True)

    ax2.scatter(p_x_gt_arr[:, 0], p_x_gt_arr[:, 1], label="x-momentum", color='b')
    ax2.scatter(p_y_gt_arr[:, 0], p_y_gt_arr[:, 1], label="y-momentum", color='r')
    ax2.set_title("Momemtum Relation - Ground-Truth")
    ax2.set_xlabel("Initial Momemtum")
    ax2.set_ylabel("Final Momentum")
    ax2.legend()
    ax2.grid(True)

    fig.savefig(os.path.join(save_dir, 'momentum_check.jpg'))

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.scatter(np.arange(p_x_preds_diff.shape[0]), p_x_preds_diff, label="x-momentum", color='b')
    ax1.scatter(np.arange(p_y_preds_diff.shape[0]), p_y_preds_diff, label="y-momentum", color='r')
    ax1.set_title("Momemtum Difference - Predictions")
    ax1.set_ylabel("Delta-Momentum")
    ax1.set_xticks([])  # Remove ticks
    ax1.set_xticklabels([])  # Remove tick labels
    ax1.legend()
    ax1.grid(True)

    print(np.sum(p_x_gt_diff))

    ax1.scatter(np.arange(p_x_gt_diff.shape[0]), p_x_gt_diff, label="x-momentum", color='b')
    ax1.scatter(np.arange(p_y_gt_diff.shape[0]), p_y_gt_diff, label="y-momentum", color='r')
    ax1.set_title("Momemtum Difference - Ground-Truth")
    ax1.set_ylabel("Delta-Momentum")
    ax1.set_xticks([])  # Remove ticks
    ax1.set_xticklabels([])  # Remove tick labels
    ax1.legend()
    ax1.grid(True)

    fig2.savefig(os.path.join(save_dir, 'momentum_diff.jpg'))

def calculate_velocity_and_momentum(preds, ground_truth, group, save_dir, context_length):
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
    
    # Extract the x, y position columns for object 1 and object 2 in both predictions and ground-truth
    preds_obj1_pos = preds[:, :, 0:2]  # Extracting x,y positions for object 1 (predictions)
    preds_obj2_pos = preds[:, :, 4:6]  # Extracting x,y positions for object 2 (predictions)
    gt_obj1_pos = ground_truth[:, :, 0:2]  # Extracting x,y positions for object 1 (ground-truth)
    gt_obj2_pos = ground_truth[:, :, 4:6]  # Extracting x,y positions for object 2 (ground-truth)
    
    # Extract the velocity columns for object 1 and object 2 in both predictions and ground-truth
    preds_obj1_vel = preds[:, :, 2:4]  # Extracting x,y velocities for object 1 (predictions)
    preds_obj2_vel = preds[:, :, 6:8]  # Extracting x,y velocities for object 2 (predictions)
    gt_obj1_vel = ground_truth[:, :, 2:4]  # Extracting x,y velocities for object 1 (ground-truth)
    gt_obj2_vel = ground_truth[:, :, 6:8]  # Extracting x,y velocities for object 2 (ground-truth)
    
    # Calculate velocity by subtracting consecutive positions (position difference) for predictions
    vel_obj1_pred = preds_obj1_pos[:, 1:, :] - preds_obj1_pos[:, :-1, :]
    vel_obj2_pred = preds_obj2_pos[:, 1:, :] - preds_obj2_pos[:, :-1, :]
    
    # Calculate velocity by subtracting consecutive positions (position difference) for ground-truth
    vel_obj1_gt = gt_obj1_pos[:, 1:, :] - gt_obj1_pos[:, :-1, :]
    vel_obj2_gt = gt_obj2_pos[:, 1:, :] - gt_obj2_pos[:, :-1, :]
    
    # Concatenate velocities for both objects (x, y velocities)
    vel_pred = np.concatenate([vel_obj1_pred, vel_obj2_pred], axis=-1)  # Shape: [batch_size, timesteps - 1, 4]
    vel_gt = np.concatenate([vel_obj1_gt, vel_obj2_gt], axis=-1)  # Shape: [batch_size, timesteps - 1, 4]
    
    if vel_pred.shape[1] % 2 != 0:
        array_2_add = np.zeros((vel_pred.shape[0], 1, vel_pred.shape[2]))
        vel_pred = np.concatenate([vel_pred, array_2_add], axis=1)
        vel_gt = np.concatenate([vel_gt, array_2_add], axis=1)

    # Average over 'group' consecutive timesteps
    if group > 1:
        # Calculate the number of new timesteps after grouping
        new_timesteps = vel_pred.shape[1] // group
        
        # Reshape and average over the group dimension
        vel_pred = vel_pred.reshape(vel_pred.shape[0], new_timesteps, group, 4).mean(axis=2)[:, :-1, :]
        vel_gt = vel_gt.reshape(vel_gt.shape[0], new_timesteps, group, 4).mean(axis=2)[:, :-1, :]
    
    # Calculate momentum difference for position velocities (already done)
    momentum_pred_diff_pos = np.sum(np.abs(vel_pred[:, 1:, :] - vel_pred[:, :-1, :]), axis=(1, 2))
    momentum_gt_diff_pos = np.sum(np.abs(vel_gt[:, 1:, :] - vel_gt[:, :-1, :]), axis=(1, 2))

    # Calculate momentum difference for velocity components (using velocity values)
    # Momentum difference for velocities is simply the change in the x and y velocity components over time
    momentum_pred_diff_vel = np.sum(np.abs(preds_obj1_vel[:, 1:, :] - preds_obj1_vel[:, :-1, :]), axis=(1, 2)) + \
                             np.sum(np.abs(preds_obj2_vel[:, 1:, :] - preds_obj2_vel[:, :-1, :]), axis=(1, 2))
    
    momentum_gt_diff_vel = np.sum(np.abs(gt_obj1_vel[:, 1:, :] - gt_obj1_vel[:, :-1, :]), axis=(1, 2)) + \
                            np.sum(np.abs(gt_obj2_vel[:, 1:, :] - gt_obj2_vel[:, :-1, :]), axis=(1, 2))

    # Plot the difference in momentum for both predictions and ground-truth
    timesteps = np.arange(1, momentum_pred_diff_pos.shape[0] + 1)
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(timesteps, momentum_pred_diff_pos, label='Momentum Difference (Pred - Position)', color='b')
    plt.plot(timesteps, momentum_gt_diff_pos, label='Momentum Difference (GT - Position)', color='r')
    plt.plot(timesteps, momentum_pred_diff_vel, label='Momentum Difference (Pred - Velocity)', color='g')
    plt.plot(timesteps, momentum_gt_diff_vel, label='Momentum Difference (GT - Velocity)', color='y')
    
    plt.xlabel('Timestep')
    plt.ylabel('Momentum Difference')
    plt.title('Difference in Momentum (Predictions vs Ground Truth) - Context={}'.\
              format(context_length))
    plt.legend()
    plt.grid(True)
    plt.show()

    fig.savefig(os.path.join(save_dir, 'momentum_check.jpg'))
    
    return vel_pred, vel_gt

'''