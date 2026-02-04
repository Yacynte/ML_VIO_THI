import numpy as np

plot_2d = False
use_gps = True
DATASET_TYPE = "urbaning"  # Options: "urbaning", "thi", "kitti_mat"

cam_poses = np.load("cam_poses.npy")
imu_poses = np.load("imu_cam_coord.npy")

# print("first and last imu pose:", imu_poses[0], imu_poses[70])
cam_poses = cam_poses[:, :]  # Align lengths if necessary
# print("first and last cam pose:", cam_poses[0], cam_poses[70])
imu_poses = imu_poses[:, :]
# --- Visualization ---
print("Plotting results...")

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    if plot_2d:
        # 2D Plot
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.plot(cam_poses[:, 0], cam_poses[:, 1], label='Camera Pose', color='blue')
        if use_gps:
            ax.plot(imu_poses[:, 0], imu_poses[:, 1], label='IMU Pose', color='red', linestyle='--')
        
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.legend()
        ax.set_title(f'Camera Pose VS 2D IMU Trajectory Estimation')
        ax.axis('equal')
        ax.grid(True)
    else:
        # 3D Plot
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot(cam_poses[:, 0], cam_poses[:, 1], cam_poses[:, 2], label='Camera Pose', color='blue')
        if use_gps:
            ax.plot(imu_poses[:, 0], imu_poses[:, 1], imu_poses[:, 2], label='Ground Truth (GPS)', color='red', linestyle='--')
        
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.set_zlabel('Up (m)')
        ax.legend()
        ax.set_title(f'IMU Trajectory Estimation ({DATASET_TYPE.upper()})')
        
        # Make axis equal aspect ratio (hack for 3D)
        max_range = np.array([cam_poses[:,0].max()-cam_poses[:,0].min(), 
                            cam_poses[:,1].max()-cam_poses[:,1].min(), 
                            cam_poses[:,2].max()-cam_poses[:,2].min()]).max() / 2.0
        
        if max_range == 0: max_range = 1.0 # Prevent singular axis
        
        mid_x = (cam_poses[:,0].max()+cam_poses[:,0].min()) * 0.5
        mid_y = (cam_poses[:,1].max()+cam_poses[:,1].min()) * 0.5
        mid_z = (cam_poses[:,2].max()+cam_poses[:,2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.show()
except ImportError as e:
    print(f"Could not import matplotlib for plotting: {e}")
except Exception as e:
    print(f"An error occurred during plotting: {e}")
