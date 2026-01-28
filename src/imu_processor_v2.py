import numpy as np
import pandas as pd
import os
import json
from scipy.spatial.transform import Rotation as R
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# --- Constants ---
GRAVITY_WORLD = np.array([0, 0, -9.81]) # World Gravity Vector (m/s^2)
# WGS-84 Earth Parameters for ENU conversion
A_EARTH = 6378137.0
F_EARTH = 1 / 298.257223563
E_SQUARED = 2 * F_EARTH - F_EARTH**2

class IMUDataLoader:
    """Handles loading of IMU and Ground Truth data."""
    def __init__(self, folder_path, dataset_type='kitti', timestamp_file=None):
        self.folder_path = folder_path
        self.dataset_type = dataset_type
        self.timestamp_file = timestamp_file
        self.data_files = []
        self.timestamps = []
        
        self._load_file_list()
        self._load_timestamps()

    def _load_file_list(self):
        if self.dataset_type == 'urbaning':
            # UrbanIng: JSON files directly in folder
            files = sorted([f for f in os.listdir(self.folder_path) if f.endswith('.json')])
            self.data_files = [os.path.join(self.folder_path, f) for f in files]
        elif self.dataset_type == 'kitti':
            # KITTI: Text files in 'data' subdirectory usually
            data_path = os.path.join(self.folder_path, 'data')
            if not os.path.exists(data_path):
                data_path = self.folder_path # Fallback if 'data' subdir doesn't exist
            files = sorted([f for f in os.listdir(data_path) if f.endswith('.txt')])
            self.data_files = [os.path.join(data_path, f) for f in files]

    def _load_timestamps(self):
        if self.dataset_type == 'urbaning':
            # UrbanIng: Timestamp is filename (ms)
            times = []
            for f in self.data_files:
                fname = os.path.basename(f).replace('.json', '')
                try:
                    ts = float(fname) / 1000.0 # Convert ms to seconds
                    times.append(ts)
                except ValueError:
                    continue
            self.timestamps = np.array(times, dtype=np.float64)
            
        elif self.dataset_type == 'kitti':
            # KITTI: Timestamps in separate file
            if self.timestamp_file and os.path.exists(self.timestamp_file):
                times = []
                with open(self.timestamp_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            # KITTI formatted dates or seconds?
                            # Usually formatted dates, but let's try parsing
                            try:
                                pd_ts = pd.to_datetime(line.strip())
                                times.append(pd_ts.timestamp())
                            except:
                                times.append(float(line.strip()))
                self.timestamps = np.array(times, dtype=np.float64)
            else:
                # Fallback: Create synthetic timestamps if missing
                print("Warning: No timestamp file found. creating synthetic timestamps (10Hz).")
                self.timestamps = np.arange(len(self.data_files)) * 0.1

    def get_data_at_index(self, index):
        if index >= len(self.data_files):
            return None
            
        filepath = self.data_files[index]
        
        if self.dataset_type == 'urbaning':
            with open(filepath, 'r') as f:
                data = json.load(f)
            # UrbanIng Format:
            # vA: accel [ax, ay, az]
            # W: gyro [wx, wy, wz]
            # vV: velocity [vx, vy, vz] (Body frame)
            # gTv: Pose of Vehicle in Ground Frame (T_GV)
            accel = np.array(data['vA']).flatten()
            gyro = np.array(data['W']).flatten()
            gt_pose = np.array(data['gTv'])
            
            gt_pos = gt_pose[:3, 3] # Vehicle Center Position
            gt_rot = gt_pose[:3, :3] # Vehicle Orientation
            
            # vV is velocity in Body frame (Forward, Left, Up)
            # Dead reckoning system expects velocity in World Frame
            gt_vel_v = np.array(data['vV']).flatten()
            gt_vel_w = gt_rot @ gt_vel_v
            
            return accel, gyro, gt_pos, gt_rot, gt_vel_w
            
        elif self.dataset_type == 'kitti':
            with open(filepath, 'r') as f:
                line = f.readline()
            vals = list(map(float, line.strip().split()))
            # KITTI OXTS Format (0-indexed):
            # 11,12,13: ax, ay, az (m/s^2)
            # 17,18,19: wx, wy, wz (rad/s)
            # 0,1,2: lat, lon, alt
            # 8,9,10: vf, vl, vu (forward, left, up velocity) - requires rotation to world
            # 3,4,5: roll, pitch, yaw
            
            accel = np.array([vals[11], vals[12], vals[13]])
            gyro = np.array([vals[17], vals[18], vals[19]])
            
            # Ground truth (GPS LLA)
            lat, lon, alt = vals[0], vals[1], vals[2]
            
            # Orientation (r, p, y)
            r, p, y = vals[3], vals[4], vals[5]
            rot = R.from_euler('xyz', [r, p, y], degrees=False).as_matrix()
            
            # Velocity (Body frame -> World frame?)
            # vals[8,9,10] are velocities in navigation frame (East, North, Up) or Body?
            # Kitti docs say: vf, vl, vu -> vel forward, vel left, vel up (Body frame)
            # We need to rotate these to world using orientation if we want world velocity GT
            # For simplicity, we'll store them as is and process later or ignore GT velocity
            vel_body = np.array([vals[8], vals[9], vals[10]])
            
            return accel, gyro, (lat, lon, alt), rot, vel_body

class DeadReckoningINS:
    def __init__(self, initial_pos, initial_vel, initial_quat):
        self.p_w = initial_pos.copy()      # Position in World
        self.v_w = initial_vel.copy()      # Velocity in World
        self.q_w = initial_quat.copy()     # Orientation (Quaternion) [x, y, z, w]
        self.last_acc_w = None             # For trapezoidal integration
        
    def propagate(self, accel_b, gyro_b, dt):
        """
        Propagate state using IMU measurements.
        accel_b: Acceleration in Body frame (m/s^2)
        gyro_b: Angular velocity in Body frame (rad/s)
        dt: Time delta (s)
        """
        
        # 1. Update Orientation (Quaternion integration)
        # q_next = q_prev * exp(0.5 * omega * dt)
        r_current = R.from_quat(self.q_w)
        
        # Using rotation vector for update (exponential map approximation)
        delta_rot = R.from_rotvec(gyro_b * dt)
        r_next = r_current * delta_rot
        self.q_w = r_next.as_quat()
        self.q_w /= np.linalg.norm(self.q_w) # Normalize
        
        # 2. Transform Acceleration to World Frame
        # a_w = R_wb * a_b + g_w
        # Note on Gravity:
        # Accelerometer measures proper acceleration: f = a - g
        # So a = f + g.
        # If g is [0, 0, -9.81] (pointing down), we add it.
        # But commonly GRAVITY_WORLD is defined as the force vector to SUBTRACT or ADD depending on convention.
        # Let's assume standard kinematics:
        # Measured `accel_b` includes reaction force to gravity (points UP when stationary).
        # We rotate `accel_b` to world, then SUBTRACT gravity magnitude (or add [0,0,-9.8]).
        # If stationary flat: a_b = [0, 0, 9.81]. R=I. a_w = [0, 0, 9.81].
        # We want a_net = [0,0,0]. So we add [0, 0, -9.81].
        
        accel_w = r_next.apply(accel_b) # Rotate body acc to world
        accel_net_w = accel_w + GRAVITY_WORLD
        
        # 3. Update Position and Velocity (Trapezoidal or Simple Euler)
        # Using Simple Euler for clarity (Trapezoidal is better but this is sufficient for basic DR)
        
        self.p_w = self.p_w + self.v_w * dt + 0.5 * accel_net_w * dt * dt
        self.v_w = self.v_w + accel_net_w * dt
        
        return self.p_w.copy(), self.v_w.copy(), self.q_w.copy()

def gps_to_enu(lat, lon, alt, lat0, lon0, alt0):
    """
    Convert LLA to ENU coordinates relative to origin.
    """
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)
    
    d_lat = lat_rad - lat0_rad
    d_lon = lon_rad - lon0_rad
    d_alt = alt - alt0
    
    W = np.sqrt(1 - E_SQUARED * np.sin(lat0_rad)**2)
    N = A_EARTH / W
    
    x = (N + alt0) * np.cos(lat0_rad) * d_lon # East
    y = (A_EARTH * (1 - E_SQUARED) / W**3 + alt0) * d_lat # North
    z = d_alt # Up
    
    return np.array([x, y, z])

def get_processed_imu_data(dataset_type='kitti', imu_folder=None, timestamp_file=None, start_index=0, end_index=None, start_time=None, end_time=None):
    """
    Returns processed IMU data and Ground Truth for external use (e.g. by VO system).
    
    Args:
        dataset_type (str): 'kitti' or 'urbaning'
        imu_folder (str): Path to IMU data directory
        timestamp_file (str): Path to timestamp file (for Kitti)
        start_index (int): Starting index for processing (ignored if start_time is set)
        end_index (int): Ending index for processing (exclusive, ignored if end_time is set)
        start_time (float): Starting timestamp (s)
        end_time (float): Ending timestamp (s)
        
    Returns:
        tuple: (imu_trajectory, gt_trajectory)
            - imu_trajectory: np.ndarray (N, 7) [roll, pitch, yaw, x, y, z, timestamp]
            - gt_trajectory: np.ndarray (N, 3) [x, y, z] (World Frame)
    """
    if imu_folder is None:
        print("Error: IMU folder must be provided.")
        return None, None
        
    loader = IMUDataLoader(imu_folder, dataset_type=dataset_type, timestamp_file=timestamp_file)
    
    num_files = len(loader.data_files)
    if num_files == 0:
        print("No data files found.")
        return None, None

    # Determine start_index from time if provided
    if start_time is not None:
        start_idx = np.abs(loader.timestamps - start_time).argmin()
    else:
        start_idx = start_index

    # Calculate final end index
    if end_time is not None:
        end_idx = np.abs(loader.timestamps - end_time).argmin() + 1
    elif end_index is None:
        end_idx = num_files
    else:
        end_idx = min(num_files, end_index)
    
    # Check bounds
    if start_idx >= num_files:
        print(f"Error: start_index {start_idx} exceeds number of files {num_files}.")
        return None, None
    if start_idx >= end_idx:
        print(f"Error: start_index {start_idx} is greater than or equal to end_idx {end_idx}.")
        return None, None

    # Arrays to store history
    traj_data = [] # Will store [roll, pitch, yaw, x, y, z, timestamp]
    traj_gt = []   # Will store [x, y, z]
    
    # --- Initialization ---
    accel0, gyro0, gt_pos0, gt_rot0, gt_vel0 = loader.get_data_at_index(start_idx)
    
    origin_lla = None
    
    # Auto-initialize from start_idx frame (Ground Truth or Default)
    if dataset_type == 'kitti':
        # Get true sequence origin (index 0) for consistent ENU coordinates
        _accel_orig, _gyro_orig, origin_lla, _rot_orig, _vel_orig = loader.get_data_at_index(0)
        
        # Initial Position (relative to sequence start)
        p0 = gps_to_enu(gt_pos0[0], gt_pos0[1], gt_pos0[2], origin_lla[0], origin_lla[1], origin_lla[2])
        q0 = R.from_matrix(gt_rot0).as_quat()
        v0 = gt_rot0 @ gt_vel0
    elif dataset_type == 'urbaning':
        p0 = gt_pos0
        q0 = R.from_matrix(gt_rot0).as_quat()
        v0 = gt_vel0
    else:
        p0 = np.zeros(3)
        q0 = R.from_euler('xyz', [0, 0, 0]).as_quat()
        v0 = np.zeros(3)
    
    imu_sys = DeadReckoningINS(p0, v0, q0)
    
    # Store initial state
    rpy0 = R.from_quat(q0).as_euler('xyz', degrees=False)
    traj_data.append(np.hstack((rpy0, p0, loader.timestamps[start_idx])))
    
    # Store initial GT
    if dataset_type == 'kitti':
        traj_gt.append(p0)
    else:
        traj_gt.append(gt_pos0)
    
    for i in range(start_idx + 1, end_idx):
        dt = loader.timestamps[i] - loader.timestamps[i-1]
        if dt <= 0: dt = 0.01
        
        accel, gyro, gt_pos, gt_rot, gt_vel = loader.get_data_at_index(i)
        
        # Propagate
        p_est, v_est, q_est = imu_sys.propagate(accel, gyro, dt)
        
        rpy_est = R.from_quat(q_est).as_euler('xyz', degrees=False)
        traj_data.append(np.hstack((rpy_est, p_est, loader.timestamps[i])))
        
        # Store GT
        if dataset_type == 'kitti':
            gt_xyz = gps_to_enu(gt_pos[0], gt_pos[1], gt_pos[2], 
                                origin_lla[0], origin_lla[1], origin_lla[2])
            traj_gt.append(gt_xyz)
        else:
            traj_gt.append(gt_pos)
    
    print(f"Processed {len(traj_data)} IMU frames (start_index={start_idx}, end_index={end_idx})")
    return np.array(traj_data), np.array(traj_gt)

def run_processing(use_gps=True, plot_2d=False):
    # --- Configuration ---
    # Edit these paths as needed
    # Urbaning Example
    # DATASET_TYPE = 'urbaning'
    # IMU_FOLDER = '/home/divan/ML_VIO_THI/datasets/UrbanIng-V2X/dataset/20241126_0010_crossing2_00/vehicle1_state'
    # TIMESTAMP_FILE = None
    
    # Kitti Example
    DATASET_TYPE = 'kitti'
    # Update this path to your actual data location
    IMU_FOLDER = '/home/divan/ML_VIO_THI/2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync/oxts' 
    TIMESTAMP_FILE = os.path.join(IMU_FOLDER, '..', 'oxts', 'timestamps.txt') # Usually here, let's check
    if not os.path.exists(TIMESTAMP_FILE):
         # Try to find it in similar location or let loader synthesize
         TIMESTAMP_FILE = '/home/divan/ML_VIO_THI/2011_09_26_drive_0013_sync/2011_09_26/2011_09_26_drive_0013_sync/oxts/timestamps.txt'
    
    # ---------------------
    
    print(f"Loading data from {IMU_FOLDER}...")
    loader = IMUDataLoader(IMU_FOLDER, dataset_type=DATASET_TYPE, timestamp_file=TIMESTAMP_FILE)
    
    if len(loader.data_files) == 0:
        print("No data files found. Please check the path.")
        return

    # Arrays to store history
    traj_est = []
    traj_gt = []
    
    # --- Initialization ---
    # --- Initialization ---
    accel0, gyro0, gt_pos0, gt_rot0, gt_vel0 = loader.get_data_at_index(0)
    
    origin_lla = None
    if use_gps:
        # Initialize state with Ground Truth at t=0
        if DATASET_TYPE == 'kitti':
            origin_lla = gt_pos0
            p0 = np.zeros(3) # Start at 0,0,0 local
            q0 = R.from_matrix(gt_rot0).as_quat()
            v0 = gt_rot0 @ gt_vel0 # Rotate to World
        else:
            p0 = gt_pos0
            q0 = R.from_matrix(gt_rot0).as_quat()
            v0 = gt_vel0
    else:
        # Manual Initialization (No GPS)
        print("Initializing with Default/Manual state (No GPS)...")
        p0 = np.zeros(3)
        # Assuming starting flat and stationary
        q0 = R.from_euler('xyz', [0, 0, 0], degrees=False).as_quat()
        v0 = np.zeros(3) 

    imu_sys = DeadReckoningINS(p0, v0, q0)
    
    traj_est.append(p0)
    if use_gps:
        traj_gt.append(p0)
    
    print(f"Processing {len(loader.data_files)} frames...")
    
    for i in range(1, len(loader.data_files)):
        dt = loader.timestamps[i] - loader.timestamps[i-1]
        if dt <= 0: dt = 0.01 # Fallback
        
        accel, gyro, gt_pos, gt_rot, gt_vel = loader.get_data_at_index(i)
        
        # Propagate State
        p_est, v_est, q_est = imu_sys.propagate(accel, gyro, dt)
        traj_est.append(p_est)
        
        # Store GT
        if use_gps:
            if DATASET_TYPE == 'kitti':
                gt_xyz = gps_to_enu(gt_pos[0], gt_pos[1], gt_pos[2], 
                                    origin_lla[0], origin_lla[1], origin_lla[2])
                traj_gt.append(gt_xyz)
            else:
                traj_gt.append(gt_pos)
            
    traj_est = np.array(traj_est)
    if use_gps:
        traj_gt = np.array(traj_gt)
    
    # --- Visualization ---
    print("Plotting results...")
    
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    
        if plot_2d:
            # 2D Plot
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            ax.plot(traj_est[:, 0], traj_est[:, 1], label='IMU Dead Reckoning', color='blue')
            if use_gps:
                ax.plot(traj_gt[:, 0], traj_gt[:, 1], label='Ground Truth (GPS)', color='red', linestyle='--')
            
            ax.set_xlabel('East (m)')
            ax.set_ylabel('North (m)')
            ax.legend()
            ax.set_title(f'2D IMU Trajectory ({DATASET_TYPE.upper()})')
            ax.axis('equal')
            ax.grid(True)
        else:
            # 3D Plot
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            
            ax.plot(traj_est[:, 0], traj_est[:, 1], traj_est[:, 2], label='IMU Dead Reckoning', color='blue')
            if use_gps:
                ax.plot(traj_gt[:, 0], traj_gt[:, 1], traj_gt[:, 2], label='Ground Truth (GPS)', color='red', linestyle='--')
            
            ax.set_xlabel('East (m)')
            ax.set_ylabel('North (m)')
            ax.set_zlabel('Up (m)')
            ax.legend()
            ax.set_title(f'IMU Trajectory Estimation ({DATASET_TYPE.upper()})')
            
            # Make axis equal aspect ratio (hack for 3D)
            max_range = np.array([traj_est[:,0].max()-traj_est[:,0].min(), 
                                traj_est[:,1].max()-traj_est[:,1].min(), 
                                traj_est[:,2].max()-traj_est[:,2].min()]).max() / 2.0
            
            if max_range == 0: max_range = 1.0 # Prevent singular axis
            
            mid_x = (traj_est[:,0].max()+traj_est[:,0].min()) * 0.5
            mid_y = (traj_est[:,1].max()+traj_est[:,1].min()) * 0.5
            mid_z = (traj_est[:,2].max()+traj_est[:,2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.show()
    except ImportError as e:
        print(f"Could not import matplotlib for plotting: {e}")
    except Exception as e:
        print(f"An error occurred during plotting: {e}")

if __name__ == "__main__":
    run_processing(use_gps=True, plot_2d=True)
