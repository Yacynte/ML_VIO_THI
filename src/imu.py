import numpy as np
import pandas as pd
from typing import List, Tuple
import scipy.integrate as spi
import os
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class IMUDataLoader:
    """Class to load and process IMU data from oxts kitti data files."""
    def __init__(self, folder_path: str, dataset_type: str = 'kitti', timestamp_file: str = None, ground_truth_file: str = None):
        self.dataset_type = dataset_type # e.g., 'kitti', 'urbaning', 'kitti_mat'
        # --- File Path Setup ---
        self.folder_path = folder_path
        if self.dataset_type == 'kitti':
            self.timestamp_file = os.path.join(folder_path, 'timestamps.txt')
            self.imu_path = os.path.join(folder_path, 'data') # Added 'data' subdirectory
        elif self.dataset_type == 'urbaning':
            self.imu_path = folder_path  # UrbanIng dataset has IMU files directly in the folder
            self.timestamp_file = None  # Timestamps will be extracted from filenames
        elif self.dataset_type == 'kitti_mat':
            self.imu_path = folder_path  # KITTI .mat files directly in the folder
            self.timestamp_file = timestamp_file  # Timestamps from provided file
            self.ground_truth_file = ground_truth_file
            self.N = 10 # default Number of IMU measurements per camera frame
        self.initial_roll = 0
        self.initial_pitch = 0.0
        self.initial_yaw = 0.0  
        self.initial_velocity = np.array([0.0, 0.0, 0.0])
        self.initial_position = np.array([0.0, 0.0, 0.0])

        # --- Data Loading and Storage ---
        # Store the list of file paths and timestamps as attributes
        self.imu_file_paths = self.__load_imu_files()
        self.timestamps = self.__load_timestamp()

        # imu_data should store the *extracted* IMU readings. 
        # Initialize as an empty list (to be filled later) or an empty array.
        self.extracted_imu_data = [] 
        
        # --- Prediction/Filter State Initialization (for processing) ---
        self.prev_timestamp = None
        self.prev_imu_data = None
        
        # --- Data Extraction Constants ---
        # ax, ay, az, wx, wy, wz (0-based indices)
        self.imu_indices = [11, 12, 13, 17, 18, 19]
        self.gps_indices = [1, 0, 2]  # lon, lat, alt (0-based indices)
        self.orientation_indices = [3, 4, 5]  # roll, pitch, yaw (0-based indices)
        self.velocity_indices = [8, 9, 10]  # vx, vy, vz (0-based indices)
        
        # --- Immediate Processing (Optional) ---
        # The line below is for demonstration/testing. 
        # You may want to iterate and load all data outside of __init__
        # self.extract_imu_data(self.imu_file_paths[0]) 

    def imu_data_generator(self):
        return self.imu_file_paths, self.timestamps
    
    
    def extract_imu_data(self, oxts_filepath: str, initial_alignment: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reads a single KITTI OXTS .txt file and extracts the IMU data.

        The IMU data corresponds to 0-based indices in the 30-value vector:
        - Linear Accelerations: indices 12, 13, 14 (ax, ay, az)
        - Angular Rates: indices 18, 19, 20 (wx, wy, wz)

        Args:
            oxts_filepath: Full path to the OXTS data file (e.g., '0000000000.txt').

        Returns:
            A tuple containing:
            - Linear accelerations (ax, ay, az) in m/s^2.
            - Angular rates (wx, wy, wz) in rad/s.
        """
        if self.dataset_type == 'urbaning':
            import json
            try:
                with open(oxts_filepath, 'r') as f:
                    data_json = json.load(f)
                # Extract IMU components
                accel = np.array([data_json['vA']]).reshape(-1)
                # Extract gyro components
                gyro = np.array([data_json['W']]).reshape(-1)
                # Extract GPS components
                initial_pose = np.array([data_json['gTv']]).squeeze(0)
                # Extract initial orientation
                rot_mat = initial_pose[:3, :3]
                r_init = R.from_matrix(rot_mat)
                rvec = r_init.as_rotvec()
                if initial_alignment:
                    self.initial_roll, self.initial_pitch, self.initial_yaw = rvec
                    # self.initial_yaw, self.initial_pitch, self.initial_roll = rvec
                    self.initial_velocity = np.array(data_json['vV'])
                    self.initial_position = initial_pose[:3, 3]
                position = initial_pose[:3, 3]
                return accel, gyro, position
            except FileNotFoundError:
                print(f"Error: File not found at {oxts_filepath}")
                return np.zeros(3), np.zeros(3), np.zeros(3)
            except ValueError as e:
                print(f"Error parsing data in {oxts_filepath}: {e}")
                return np.zeros(3), np.zeros(3), np.zeros(3)

        if self.dataset_type == 'kitti_mat':
            from scipy.io import loadmat
            try:
                data = loadmat(oxts_filepath)
                imu_data = data['imu_data_interp']
                accel = imu_data[:, 0:3]
                gyro = imu_data[:, 3:6]
                # Each line has 12 floats → automatically shaped (num_lines, 12)
                data = np.loadtxt(self.ground_truth_file)
                # reshape to (num_poses, 3, 4)
                poses = data.reshape(-1, 3, 4)
                position = poses[:, :, 3]  # shape (num_poses, 3) Assuming columns 10,11,12 are x,y,z
                # extract rotation matrices (3x3)
                rot_mats = poses[:, :, :3]  # shape (num_poses, 3, 3)
                # vectorized conversion
                rot_vecs = R.from_matrix(rot_mats).as_rotvec()  # shape (num_poses, 3)
                if initial_alignment:
                    self.initial_roll, self.initial_pitch, self.initial_yaw = rot_vecs[0]
                    # self.initial_velocity = imu_data[6:9, 0]
                    self.initial_position = position[0]
                return accel, gyro, position
            except FileNotFoundError:
                print(f"Error: File not found at {oxts_filepath}")
                return np.zeros(3), np.zeros(3), np.zeros(3)
            except ValueError as e:
                print(f"Error parsing data in {oxts_filepath}: {e}")
                return np.zeros(3), np.zeros(3), np.zeros(3)

        try:
            # Read the single line from the file
            with open(oxts_filepath, 'r') as f:
                data_line = f.readline()

            # Split by space and convert all 30 values to floats
            # This will raise a ValueError if the data isn't clean
            values = [float(val) for val in data_line.split()]

            if len(values) != 30:
                raise ValueError(f"Expected 30 values, but found {len(values)} in the file.")

            # 4. Extract the relevant IMU components
            imu_components = np.array([values[i] for i in self.imu_indices])

            # 5. Extract GPS components
            gps_components = np.array([values[i] for i in self.gps_indices])

            # 6. (Optional) Extract orientation components (roll, pitch, yaw)
            if initial_alignment:
                self.initial_roll = values[self.orientation_indices[0]] 
                self.initial_pitch = values[self.orientation_indices[1]] 
                self.initial_yaw = values[self.orientation_indices[2]]   
                self.initial_velocity = np.array([values[i] for i in self.velocity_indices])

            # imu_orientation = np.array([values[i] for i in self.orientation_indices])
            # Split into accelerations and angular rates
            accel = imu_components[:3]  # indices 0, 1, 2 of the 6-vector
            gyro = imu_components[3:]   # indices 3, 4, 5 of the 6-vector

            return accel, gyro, gps_components

        except FileNotFoundError:
            print(f"Error: File not found at {oxts_filepath}")
            return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
        except ValueError as e:
            print(f"Error parsing data in {oxts_filepath}: {e}")
            return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
        
    def __load_imu_files(self):
        """Loads IMU data from all OXTS files in the specified folder."""
        imu_files_list = []
        if self.dataset_type == 'urbaning':
            for filename in sorted(os.listdir(self.imu_path)):
                if filename.endswith('.json'):
                    imu_files_list.append(filename)
            return imu_files_list
        if self.dataset_type == 'kitti_mat':
            filename = os.path.basename(self.imu_path)
            if filename.endswith('.mat'):
                imu_files_list.append(filename)
            return imu_files_list
        for filename in sorted(os.listdir(self.imu_path)):
            if filename.endswith('.txt'):
                imu_files_list.append(filename)
        return imu_files_list
    

    def __load_timestamp(self):
        """Loads timestamp data from all OXTS files in the specified folder."""
        timestamps = []

        if self.dataset_type == 'urbaning':
            for filename in sorted(os.listdir(self.imu_path)):
                if filename.endswith('.json'):
                    timestamp_str = filename.replace('.json', '')
                    pd_timestamp = pd.to_datetime(int(timestamp_str), unit="ms")
                    timestamp_sec = pd_timestamp.timestamp()
                    timestamps.append(timestamp_sec)
            timestamps = np.array(timestamps, dtype=np.float64)
            return timestamps
        
        if self.dataset_type == 'kitti_mat':
            with open(self.timestamp_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:  # skip empty lines
                        timestamps.append(float(line))
            cam_times = np.array(timestamps, dtype=np.float64)

            # cam_times: shape (num_cam_frames,)
            # N: imu frequency / camera frequency
            imu_times = []

            for i in range(len(cam_times) - 1):
                start = cam_times[i]
                end = cam_times[i + 1]
                # N measurements evenly spaced between start and end (excluding start)
                imu_times.extend(np.linspace(start, end, self.N, endpoint=False))

            # Optionally add the last frame's times
            imu_times.append(cam_times[-1])
            timestamps = np.array(imu_times)
            return timestamps

        with open(self.timestamp_file, 'r') as f:
            for line in f:
                timestamps_str = line.strip()
                pd_timestamp = pd.to_datetime(timestamps_str)
                timestamp_sec = pd_timestamp.timestamp()
                timestamps.append(timestamp_sec)
        timestamps = np.array(timestamps, dtype=np.float64)
        # timestamps = timestamps - timestamps[0]  # Normalize to start at zero
        return timestamps
    
# WGS-84 Earth Parameters (Constants for ECEF to Local Frame Conversion)
A = 6378137.0               # Semi-major axis (equatorial radius) in meters
F = 1 / 298.257223563       # Flattening
E_SQUARED = 2 * F - F**2    # First eccentricity squared

def convert_to_local_enu(lon_deg: float, lat_deg: float, alt: float, lon_0_deg: float, 
                         lat_0_deg: float, alt_0: float) -> np.ndarray:
    """
    Converts a single GPS reading (LLA) to local East-North-Up (ENU) coordinates (meters) 
    relative to a specific origin point (lon_0, lat_0, alt_0).

    This uses the simplified Local Tangent Plane (LTP) projection.

    Args:
        lon_deg (float): Longitude of the current point (degrees).
        lat_deg (float): Latitude of the current point (degrees).
        alt (float): Altitude of the current point (meters).
        lon_0_deg (float): Longitude of the origin (degrees).
        lat_0_deg (float): Latitude of the origin (degrees).
        alt_0 (float): Altitude of the origin (meters).

    Returns:
        np.ndarray: Local position vector [East, North, Up] in meters.
    """

    # Convert origin and current latitude to radians for trigonometric functions
    lon_0_rad = np.deg2rad(lon_0_deg)
    lat_0_rad = np.deg2rad(lat_0_deg)

    lon_rad = np.deg2rad(lon_deg)
    lat_rad = np.deg2rad(lat_deg)

    # Calculate radii of curvature at the origin latitude (lat_0)
    W_0 = np.sqrt(1 - E_SQUARED * np.sin(lat_0_rad)**2)
    
    # Meridian Radius of Curvature (R_N)
    R_N0 = A * (1 - E_SQUARED) / W_0**3
    
    # Prime Vertical Radius of Curvature (R_E)
    R_E0 = A / W_0

    # Calculate differences in LLA
    d_lon = lon_rad - lon_0_rad
    d_lat = lat_rad - lat_0_rad
    d_alt = alt - alt_0

    # Apply the Local Tangent Plane (LTP) approximation:
    
    # East (X) component (change in longitude scaled by the local radius)
    # E = (R_E0 + H_0) * d_lon * cos(lat_0)
    E = (R_E0 + alt_0) * d_lon * np.cos(lat_0_rad)
 
    # North (Y) component (change in latitude scaled by the local radius)
    # N = (R_N0 + H_0) * d_lat
    N = (R_N0 + alt_0) * d_lat 
    
    # Up (Z) component (simple altitude difference)
    U = d_alt

    # Return the local position vector [East, North, Up]
    return np.array([E, N, U]) # Note: ENU order
        

def get_imu_data(imu_folder: str = '/path/to/imu/data/folder', dataset_type: str = 'kitti', timestamp_file: str = None, ground_truth_file: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    imuLoader = IMUDataLoader(imu_folder, dataset_type=dataset_type, timestamp_file=timestamp_file, ground_truth_file=ground_truth_file)
    imu_files, timestamps = imuLoader.imu_data_generator()
    imu_data = []
    gps_coords = []
    full_path = os.path.join(imuLoader.imu_path, imu_files[0])
    if dataset_type == 'kitti_mat':
        accel, gyro, gps = imuLoader.extract_imu_data(imuLoader.imu_path, initial_alignment=True)
        initial_orientation = (imuLoader.initial_roll, imuLoader.initial_pitch, imuLoader.initial_yaw)
        initial_velocity = imuLoader.initial_velocity
        initial_position = imuLoader.initial_position
        imu_data = np.hstack((accel, gyro, timestamps.reshape(-1, 1)))
        gps_coords = gps
        return imu_data, gps_coords, np.array(initial_orientation), initial_velocity, initial_position
    _, _, _ = imuLoader.extract_imu_data(full_path, initial_alignment=True)
    initial_orientation = (imuLoader.initial_roll, imuLoader.initial_pitch, imuLoader.initial_yaw)
    initial_velocity = imuLoader.initial_velocity
    initial_position = imuLoader.initial_position
    for imu_file, timestamp in zip(imu_files, timestamps):
        full_path = os.path.join(imuLoader.imu_path, imu_file)
        accel, gyro, gps = imuLoader.extract_imu_data(full_path)
        imu_data.append(np.hstack((accel, gyro, timestamp)))
        gps_coords.append(gps)

    imu_data = np.array(imu_data, dtype=np.float64)
    gps_coords = np.array(gps_coords)
    return imu_data, gps_coords, np.array(initial_orientation), initial_velocity, initial_position


def get_pose_ins(imu_data: np.ndarray, initial_position = np.array([0, 0, 0]), initial_orientation = np.array([0, 0, 0]), 
                 initial_velocity = np.array([0, 0, 0])) -> np.ndarray:
    # Global constant for gravity in the Z-up World Frame
    GRAVITY_WORLD = np.array([0, 0, 9.81]) 
    """
    Calculates Pose (Position and Orientation) from IMU data using a simple 
    dead-reckoning INS loop.

    IMU data is assumed to have shape (N, 7): [ax, ay, az, wx, wy, wz, timestamp]
    This function assumes zero sensor biases (ba=0, bg=0) and an initial level 
    orientation (R_0 = Identity).
    """
    N = imu_data.shape[0]
    # initial_orientation = np.zeros(3)
    # initial_orientation = np.array([0.0, 0, np.pi/4])  # Initial orientation in radians (Roll, Pitch, Yaw)
    
    # Extract data columns
    accel_b = imu_data[:, 0:3]  # Linear acceleration in Body frame (ax, ay, az)
    omega_b = imu_data[:, 3:6]  # Angular rate in Body frame (wx, wy, wz)
    # orientation_b = imu_data[:, 6:9]  # Orientation (roll, pitch, yaw)
    timestamps = imu_data[:, -1] # Timestamps

    # Initialize state storage
    positions_w = np.zeros((N, 3))
    velocities_w = np.zeros((N, 3))
    quaternions_w = np.zeros((N, 4))
    # Initial Orientation from provided initial_orientation (Roll, Pitch, Yaw)
    r_init = R.from_euler('xyz', initial_orientation.copy(), degrees=False)
    rvec = r_init.as_rotvec()
    print(f"Initial Orientation in degrees (Roll, Pitch, Yaw): {np.rad2deg(initial_orientation)}")
    quaternions_w[0] = r_init.as_quat()  # [x, y, z, w]
    positions_w[0] = initial_position
    # quaternions_w[0] = np.array([0, 0, 0, 1.0]) # Start with Identity quaternion [x, y, z, w]

    # # State variables for the loop
    # # Alternative: Initialize orientation from first IMU reading
    # q_prev_ = R.from_euler('xyz', orientation_b[0], degrees=False)    # Roll-Pitch-Yaw
    # q_prev = q_prev_.as_quat()
    # quaternions_w[0] = q_prev

    q_prev = quaternions_w[0].copy()
    velocities_w[0] = initial_velocity
    v_prev = velocities_w[0]
    p_prev = positions_w[0]
    
    for i in range(1, N):
        dt = timestamps[i] - timestamps[i-1]
        
        # --- 1. Orientation Update (Quaternions) ---
        # Trapezoidal rule: use average angular rate
        omega_avg = (omega_b[i] + omega_b[i-1]) / 2.0
        # omega_avg = omega_b[i]
        
        # Exponential map: map rotation vector (omega_avg * dt) to a quaternion
        delta_q = R.from_rotvec(omega_avg * dt).as_quat()
        
        # Compose previous orientation with delta rotation
        r_prev = R.from_quat(q_prev)
        r_curr = r_prev * R.from_quat(delta_q)
        q_curr = r_curr.as_quat()
        q_curr = q_curr / np.linalg.norm(q_curr)  # renormalize

        # # Alternatively, directly use the orientation from IMU data
        # q_curr_ = R.from_euler('xyz', orientation_b[i], degrees=False)    # Roll-Pitch-Yaw
        # q_curr = q_curr_.as_quat()
        # r_curr = R.from_quat(q_curr)
        # r_prev = R.from_quat(q_prev)

        quaternions_w[i] = q_curr
        
        # --- 2. Velocity and Position Update ---
        
        # Get rotation matrices (Body to World) for start and end of interval
        R_prev = r_prev.as_matrix()
        # R_curr = R.from_quat(q_curr).as_matrix()
        R_curr = r_curr.as_matrix()
        
        # Transform accelerations (specific force) into the World frame
        # NOTE: If using an INS loop, it's better to use an average rotation matrix,
        # but for simplicity, we rotate the endpoints and average the result.
        a_w_prev = R_prev @ accel_b[i-1]
        a_w_curr = R_curr @ accel_b[i]
        # a_w_prev = accel_b[i-1]
        # a_w_curr = accel_b[i]
        
        # Subtract World Gravity Vector (Compensation)
        a_actual_w_prev = a_w_prev - GRAVITY_WORLD
        a_actual_w_curr = a_w_curr - GRAVITY_WORLD

        # Average World Acceleration for integration (Trapezoidal Rule)
        a_avg_w = (a_actual_w_prev + a_actual_w_curr) / 2.0
        # a_avg_w = a_actual_w_curr
        
        # Velocity Update: v_curr = v_prev + a_avg * dt
        v_curr = v_prev + a_avg_w * dt
        velocities_w[i] = v_curr
        
        # Position Update: p_curr = p_prev + v_avg * dt
        # v_avg = (v_prev + v_curr) / 2.0
        p_curr = p_prev + v_prev * dt + 0.5 * a_avg_w * dt * dt
        # p_curr += p_prev
        positions_w[i] = p_curr
        
        # Update previous states for the next iteration
        q_prev = q_curr
        v_prev = v_curr
        p_prev = p_curr

    # Final Output Formatting: Convert Quaternions to Euler angles (Roll, Pitch, Yaw)
    angles_rpy = R.from_quat(quaternions_w).as_euler('xyz', degrees=False) # Roll-Pitch-Yaw
    
    
    # Returns a stacked array: [Roll, Pitch, Yaw, X, Y, Z, timestamp]
    # np.save('imu_positions.npy', np.hstack((positions_w[:, [2, 1, 0]], timestamps.reshape(-1, 1))))
    return np.hstack((angles_rpy, positions_w, timestamps.reshape(-1, 1)))

def get_gps_coords(gps_coords: np.ndarray) -> np.ndarray:
    """
    Converts GPS coordinates (lon, lat, alt) to local ENU coordinates 
    relative to the first GPS reading as origin.
    """
    lon_0, lat_0, alt_0 = gps_coords[0]
    local_positions = []
    # local_positions = [np.array([0.0, 0.0, 0.0])]
    for lon, lat, alt in gps_coords:
        local_pos = convert_to_local_enu(lon, lat, alt, lon_0, lat_0, alt_0)
        local_positions.append(local_pos)
    return np.array(local_positions)

    # for i in range(1, gps_coords.shape[0]):
    #     lon_prev, lat_prev, alt_prev = gps_coords[i-1]
    #     lon, lat, alt = gps_coords[i]
    #     local_pos = convert_to_local_enu(lon, lat, alt, lon_prev, lat_prev, alt_prev)
    #     local_positions.append(local_pos + local_positions[-1])
    # return np.array(local_positions)
    

def plot_imu_gps_2d(pose_ins: np.ndarray, gps_pose: np.ndarray = None):
    dist = pose_ins[-1, :] - pose_ins[0, :]
    print(f"Total Distance Travelled (IMU): {dist} meters")
    plt.figure()
    plt.plot(pose_ins[:, 0], pose_ins[:, 1], label='IMU Trajectory')
    if gps_pose is not None:
        plt.plot(gps_pose[:, 0], gps_pose[:, 1], label='GPS Trajectory', linestyle='--')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('2D IMU Trajectory from INS')
    plt.legend()
    plt.axis('equal')
    plt.grid()
    plt.show()

def plot_imu_gps_3d(pose_ins: np.ndarray, gps_pose: np.ndarray = None):
    """Plot 3D trajectory from IMU poses and GPS poses."""
    # pose_ins = (np.linalg.inv(R_cam_imu) @ pose_ins[:, :3].T).T  # Transform camera to imu frame
    dist = pose_ins[-1] - pose_ins[0]
    print(f"Total Distance Travelled (IMU): {dist} meters")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pose_ins[:, 0], pose_ins[:, 1], pose_ins[:, 2], label='IMU Trajectory')
    if gps_pose is not None:
        ax.plot(gps_pose[:, 0], gps_pose[:, 1], gps_pose[:, 2], label='GPS Trajectory', linestyle='--')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('3D IMU Trajectory from INS')
    ax.legend()
    plt.show()

def load_imu_positions(imu_path: str, dataset_type: str = 'kitti', timestamp_file: str = None, ground_truth_file: str = None) -> Tuple[np.ndarray, np.ndarray]:
    imu_data, gps_coords, initial_orientation, initial_velocity, initial_position = get_imu_data(imu_path, dataset_type=dataset_type, timestamp_file=timestamp_file, ground_truth_file=ground_truth_file)
    pose_ins = get_pose_ins(imu_data, initial_position, initial_orientation, initial_velocity)
    if dataset_type == 'urbaning' or dataset_type == 'kitti_mat':
        gps_pose = gps_coords  # Already in local frame
    else:
        gps_pose = get_gps_coords(gps_coords)
    # points_rotated = (Rot @ points.T).T 
    return pose_ins, gps_pose

def accumulate_imu_positions(imu_data: np.ndarray) -> np.ndarray:
    for i in range(1, imu_data.shape[0]):
        imu_data[i, 3:6] += imu_data[i-1, 3:6]
    return imu_data

def main():
    folder_number = 1
    # imu_folder = f'/home/divan/ML_VIO_THI/Visual-Selective-VIO/data/imus/{folder_number:02d}.mat'
    timestamp_file = f'/home/divan/ML_VIO_THI/Visual-Selective-VIO/data/sequences/{folder_number:02d}/times.txt'
    ground_truth_file = f'/home/divan/ML_VIO_THI/Visual-Selective-VIO/data/poses/{folder_number:02d}.txt'
    dataset_type = 'kitti'  # 'kitti', 'urbaning', 'kitti_mat'
    imu_folder = '2011_09_26_drive_0013_sync/2011_09_26/2011_09_26_drive_0013_sync/oxts'  
    # imu_folder = f"datasets/UrbanIng-V2X/dataset/20241126_00{folder_number:02d}_crossing2_00/vehicle1_state"
    imu_data, gps_coords, initial_orientation, initial_velocity, initial_position = get_imu_data(imu_folder, dataset_type=dataset_type, timestamp_file=timestamp_file, ground_truth_file=ground_truth_file)
    pose_ins = get_pose_ins(imu_data, initial_position, initial_orientation, initial_velocity)
    # pose_ins = accumulate_imu_positions(pose_ins)
    if dataset_type == 'urbaning' or dataset_type == 'kitti_mat':
        gps_pose = gps_coords  # Already in local frame
    else:
        gps_pose = get_gps_coords(gps_coords)
    # np.save('gps_coords.npy', gps_coords)
    # plot_imu_trajectory(pose_ins, gps_pose)
    # plot_imu_gps_2d(pose_ins[3:6], gps_pose)
    # points = pose_ins[:, 3:6]
    # rotation = R.from_euler('xyz', initial_orientation.copy(), degrees=False)
    # Rot = rotation.as_matrix()
    # points_rotated = (Rot @ points.T).T 
    # points_rotated[:, 2] *= -1  # Invert Z axis if needed
    # gps_rotated = (Rot @ gps_pose.T).T
    plot_imu_gps_3d(pose_ins[3:6], gps_pose)
    # plot_imu_gps_2d(np.rad2deg(pose_ins[:, 1:3]))


if __name__ == "__main__":
    main()
