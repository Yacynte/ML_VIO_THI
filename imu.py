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
    def __init__(self, folder_path: str):
        
        # --- File Path Setup ---
        self.folder_path = folder_path
        self.timestamp_file = os.path.join(folder_path, 'timestamps.txt')
        self.imu_path = os.path.join(folder_path, 'data') # Added 'data' subdirectory
        self.initial_roll = 0
        self.initial_pitch = 0.0
        self.initial_yaw = 0.0  

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

            
            # Split into accelerations and angular rates
            accel = imu_components[:3]  # indices 0, 1, 2 of the 6-vector
            gyro = imu_components[3:]   # indices 3, 4, 5 of the 6-vector
            
            return accel, gyro, gps_components

        except FileNotFoundError:
            print(f"Error: File not found at {oxts_filepath}")
            return np.zeros(3), np.zeros(3), np.zeros(3)
        except ValueError as e:
            print(f"Error parsing data in {oxts_filepath}: {e}")
            return np.zeros(3), np.zeros(3), np.zeros(3)


    def __load_imu_files(self):
        """Loads IMU data from all OXTS files in the specified folder."""
        imu_files_list = []
        for filename in sorted(os.listdir(self.imu_path)):
            if filename.endswith('.txt'):
                imu_files_list.append(filename)
        return imu_files_list
    

    def __load_timestamp(self):
        """Loads timestamp data from all OXTS files in the specified folder."""
        timestamps = []
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
    # W is a common term used in geodetic calculations
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
    return np.array([E, N, U])
        

def get_imu_data(imu_folder: str = '/path/to/imu/data/folder') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    imuLoader = IMUDataLoader(imu_folder)
    imu_files, timestamps = imuLoader.imu_data_generator()
    imu_data = []
    gps_coords = []
    full_path = os.path.join(imuLoader.imu_path, imu_files[0])
    _, _, _ = imuLoader.extract_imu_data(full_path, initial_alignment=True)
    initial_orientation = (imuLoader.initial_roll, imuLoader.initial_pitch, imuLoader.initial_yaw)

    for imu_file, timestamp in zip(imu_files, timestamps):
        full_path = os.path.join(imuLoader.imu_path, imu_file)
        accel, gyro, gps = imuLoader.extract_imu_data(full_path)
        imu_data.append(np.hstack((accel, gyro, timestamp)))
        gps_coords.append(gps)

    imu_data = np.array(imu_data, dtype=np.float64)
    gps_coords = np.array(gps_coords)
    return imu_data, gps_coords, np.array(initial_orientation)


def get_pose_ins(imu_data: np.ndarray, initial_orientation = np.array([0, 0, 0])) -> np.ndarray:
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
    
    # Extract data columns
    accel_b = imu_data[:, 0:3]  # Linear acceleration in Body frame (ax, ay, az)
    omega_b = imu_data[:, 3:6]  # Angular rate in Body frame (wx, wy, wz)
    timestamps = imu_data[:, -1] # Timestamps

    # Initialize state storage
    positions_w = np.zeros((N, 3))
    velocities_w = np.zeros((N, 3))
    quaternions_w = np.zeros((N, 4))
    # Initial Orientation from provided initial_orientation (yaw, pitch, roll)
    r_init_ = R.from_euler('xyz', initial_orientation, degrees=False)    # Yaw-Pitch-Roll
    R_mat = np.linalg.inv(r_init_.as_matrix())
    r_init = R.from_matrix(R_mat)
    rvec = r_init.as_rotvec()
    print(f"Initial Orientation in degrees (Roll, Pitch, Yaw): {np.rad2deg(rvec)}")
    quaternions_w[0] = r_init_.as_quat()  # [x, y, z, w]
    # quaternions_w[0] = np.array([0, 0, 0, 1.0]) # Start with Identity quaternion [x, y, z, w]

    # State variables for the loop
    q_prev = quaternions_w[0]
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
        positions_w[i] = p_curr
        
        # Update previous states for the next iteration
        q_prev = q_curr
        v_prev = v_curr
        p_prev = p_curr

    # Final Output Formatting: Convert Quaternions to Euler angles (Roll, Pitch, Yaw)
    angles_rpy = R.from_quat(quaternions_w).as_euler('xyz', degrees=False) # Yaw-Pitch-Roll
    
    # Returns a stacked array: [Yaw, Pitch, Roll, X, Y, Z]
    return np.hstack((angles_rpy, positions_w))

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
    

def plot_imu_trajectory(pose_ins: np.ndarray, gps_pose: np.ndarray = None):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pose_ins[:, 3], pose_ins[:, 4], pose_ins[:, 5], label='IMU Trajectory')
    if gps_pose is not None:
        ax.plot(gps_pose[:, 0], gps_pose[:, 1], gps_pose[:, 2], label='GPS Trajectory', linestyle='--')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('3D IMU Trajectory from INS')
    ax.legend()
    plt.show()


def plot_imu_gps_2d(pose_ins: np.ndarray, gps_pose: np.ndarray = None):
    dist = np.linalg.norm(pose_ins[-1, 3:6] - pose_ins[0, 3:6])
    print(f"Total Distance Travelled (IMU): {dist:.2f} meters")
    plt.figure()
    plt.plot(pose_ins[:, 3], pose_ins[:, 4], label='IMU Trajectory')
    if gps_pose is not None:
        plt.plot(gps_pose[:, 0], gps_pose[:, 1], label='GPS Trajectory', linestyle='--')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('2D IMU Trajectory from INS')
    plt.legend()
    plt.axis('equal')
    plt.grid()
    plt.show()

def main():
    imu_folder = '2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync/oxts'  
    imu_data, gps_coords, initial_orientation = get_imu_data(imu_folder)
    pose_ins = get_pose_ins(imu_data)
    gps_pose = get_gps_coords(gps_coords)
    # plot_imu_trajectory(pose_ins, gps_pose)
    plot_imu_gps_2d(pose_ins, gps_pose)


if __name__ == "__main__":
    main()
