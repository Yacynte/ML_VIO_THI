import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import os


class VisualOdometry:
    def __init__(self, intrinsic_matrix, feature_detector='ORB', matcher='BF', R_cam2_imu=None, t_cam2_imu=None):
        self.K = intrinsic_matrix
        self.orb = cv2.ORB_create(2000)
        self.sift = cv2.SIFT_create()
        self.akaze = cv2.AKAZE_create()
        self.shi_tomasi_params = dict(maxCorners=2000, qualityLevel=0.01, minDistance=7, blockSize=7)
        self.fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.flann_index_kdtree = 1
        index_params = dict(algorithm=self.flann_index_kdtree, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        self.method = feature_detector
        self.matcher = matcher
        self.prev_image = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.cam2_imu_R = R_cam2_imu
        self.cam2_imu_t = t_cam2_imu
        self.pose = np.eye(4)

    def __feature_detection(self, image):
        if self.method == 'ORB':
            keypoints, descriptors = self.orb.detectAndCompute(image, None)
        elif self.method == 'SIFT':
            keypoints, descriptors = self.sift.detectAndCompute(image, None)
        elif self.method == 'AKAZE':
            keypoints, descriptors = self.akaze.detectAndCompute(image, None)
        elif self.method == 'Shi-Tomasi':
            corners = cv2.goodFeaturesToTrack(image, mask=None, **self.shi_tomasi_params)
            keypoints = [cv2.KeyPoint(x=float(c[0][0]), y=float(c[0][1]), _size=7) for c in corners]
            keypoints, descriptors = self.orb.compute(image, keypoints)
        elif self.method == 'FAST':
            keypoints = self.fast.detect(image, None)
            keypoints, descriptors = self.orb.compute(image, keypoints)
        else:
            raise ValueError(f"Unknown feature detection method: {self.method}")
        return keypoints, descriptors
    
    def __feature_matching(self, desc1, desc2, prevImg = None, currImg = None, prevPts = None):
        if self.matcher == 'BF':
            matches = self.bf.match(desc1, desc2)
            matches = sorted(matches, key=lambda x: x.distance)
        elif self.matcher == 'FLANN':
            matches = self.flann.knnMatch(desc1, desc2, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            matches = good_matches
        elif self.matcher == 'LK':
            currPts, status, _ = cv2.calcOpticalFlowPyrLK(prevImg, currImg, prevPts, None, **self.lk_params)
            matches = []
            for i, st in enumerate(status):
                if st:
                    match = cv2.DMatch(_queryIdx=i, _trainIdx=i, _imgIdx=0, _distance=0)
                    matches.append(match)
        else:
            raise ValueError(f"Unknown matcher type: {self.matcher}")
        return matches

    def process_frame(self, image, scale=np.array([1.0, 1.0, 1.0])):
        cur_keypoints, cur_descriptors = self.__feature_detection(image)
        if self.prev_image is None:
            self.prev_image = image
            self.prev_keypoints = cur_keypoints
            self.prev_descriptors = cur_descriptors
            return self.pose
        prev_pts = np.array([kp.pt for kp in self.prev_keypoints], dtype=np.float32)
        matches = self.__feature_matching(self.prev_descriptors, cur_descriptors, self.prev_image, image, prev_pts)
        if len(matches) < 8:
            print("Not enough matches found!")
            return self.pose
        src_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([cur_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        E, mask = cv2.findEssentialMat(dst_pts, src_pts, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R_mat, t_vec, mask_pose = cv2.recoverPose(E, dst_pts, src_pts, self.K)
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = t_vec.flatten()
        self.pose = self.pose @ np.linalg.inv(T)
        self.prev_image = image
        self.prev_keypoints = cur_keypoints
        self.prev_descriptors = cur_descriptors

        pose_unscaled = self.pose.copy()
        pose_cam2_imu = np.eye(4)
        if self.cam2_imu_R is not None and self.cam2_imu_t is not None:
            pose_cam2_imu[:3, :3] = self.cam2_imu_R
            pose_cam2_imu[:3, 3] = self.cam2_imu_t.flatten()
            pose_imu = pose_cam2_imu @ pose_unscaled
            pose_imu[:3, 3] = pose_imu[:3, 3] * scale
        return pose_imu
    

class DataLoader:
    def __init__(self, image_folder, imu_folder):
        self.image_folder = os.path.join(image_folder, 'data')
        self.image_timestamp_file = os.path.join(image_folder, 'timestamps.txt')
        self.imu_folder = os.path.join(imu_folder, 'data')
        self.imu_timestamp_file = os.path.join(imu_folder, 'timestamps.txt')
        self.calibration_file = os.path.join(imu_folder, 'timestamps.txt')

        self.image_timestamps = self.__load_timestamp(self.image_timestamp_file)
        self.imu_timestamps = self.__load_timestamp(self.imu_timestamp_file)
        self.image_files = self.__load_image_file()
        self.imu_files = self.__load_imu_files()

        self.imu_indices = [12, 13, 14, 18, 19, 20]  # Indices for ax, ay, az, wx, wy, wz
        self.orientation_indices = [3, 4, 5]  # Indices for roll, pitch, yaw
        self.initial_roll = 0.0
        self.initial_pitch = 0.0
        self.initial_yaw = 0.0
        self.K = self.__load_calibration()
        _, _ = self.extract_imu_data(self.imu_files[0], initial_alignment = True)
        self.initial_orientation = np.array([self.initial_roll, self.initial_pitch, self.initial_yaw])
    
    def __load_timestamp(self, timestamp_file):
        """Loads timestamp data from all files in the specified folder."""
        timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f:
                timestamps_str = line.strip()
                pd_timestamp = pd.to_datetime(timestamps_str)
                timestamp_sec = pd_timestamp.timestamp()
                timestamps.append(timestamp_sec)
        timestamps = np.array(timestamps)
        timestamps = timestamps - timestamps[0]  # Normalize to start at zero
        return timestamps
    
    def __load_imu_files(self):
        """Loads IMU data from all OXTS files in the specified folder."""
        imu_files_list = []
        for filename in sorted(os.listdir(self.imu_folder)):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.imu_folder, filename)
                imu_files_list.append(filepath)
        return imu_files_list
    
    def __load_calibration(self):
        # Read the single line from the file
        with open(self.calibration_file, 'r') as f:
            data_line = f.readline()
        calibration_indices = []
        # Split by space and convert all 30 values to floats
        # This will raise a ValueError if the data isn't clean
        values = [float(val) for val in data_line.split()]

        if len(values) != 30:
            raise ValueError(f"Expected 30 values, but found {len(values)} in the file.")

        # 4. Extract the relevant IMU components
        intrinsic = np.array([values[i] for i in calibration_indices])

        k = intrinsic.reshape(3,3)

        return k
    
    def __load_image_file(self):
        image_files = sorted([f for f in os.listdir(self.image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
        image_paths = [os.path.join(self.image_folder, f) for f in image_files]
        return image_paths
    
    def extract_imu_data(self, oxts_filepath: str, initial_alignment: bool = False):
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

            # 6. (Optional) Extract orientation components (roll, pitch, yaw)
            if initial_alignment:
                self.initial_roll = values[self.orientation_indices[0]] 
                self.initial_pitch = values[self.orientation_indices[1]] 
                self.initial_yaw = values[self.orientation_indices[2]]   

            
            # Split into accelerations and angular rates
            accel = imu_components[:3]  # indices 0, 1, 2 of the 6-vector
            gyro = imu_components[3:]   # indices 3, 4, 5 of the 6-vector
            
            return accel, gyro

        except FileNotFoundError:
            print(f"Error: File not found at {oxts_filepath}")
            return np.zeros(3), np.zeros(3)
        except ValueError as e:
            print(f"Error parsing data in {oxts_filepath}: {e}")
            return np.zeros(3), np.zeros(3)


class IMUIntegrator:
    def __init__(self, initial_orientation = np.array([0.0, 0.0, 0.0])):
        # self.imu_data = imu_data  # Expected shape: (N, 7) -> [timestamp, ax, ay, az, gx, gy, gz]
        self.gravity = np.array([0, 0, 9.81])  # Gravity vector in world frame
        self.initial_orientation = initial_orientation  # Roll, Pitch, Yaw
    
    def integrate(self, imu_data: np.ndarray):
        N = imu_data.shape[0]
        positions_w = np.zeros((N, 3))
        velocities_w = np.zeros((N, 3))
        quaternions_w = np.zeros((N, 4))
        # Initial Orientation from provided initial_orientation (roll, pitch, yaw)
        r_init = R.from_euler('xyz', self.initial_orientation, degrees=False)    # Roll-Pitch-Yaw
        quaternions_w[0] = r_init.as_quat()  # [x, y, z, w]
        # quaternions_w[0] = np.array([0, 0, 0, 1.0]) # Start with Identity quaternion [x, y, z, w]

        for i in range(1, N):
            dt = imu_data[i, 0] - imu_data[i - 1, 0]
            acc_body = imu_data[i - 1, 1:4]
            gyro_body = imu_data[i - 1, 4:7]

            # --- 1. Orientation Update ---
            q_prev = quaternions_w[i - 1]
            omega = gyro_body
            omega_norm = np.linalg.norm(omega)
            if omega_norm > 1e-6:
                theta = omega_norm * dt
                axis = omega / omega_norm
                delta_q = R.from_rotvec(axis * theta).as_quat()
            else:
                delta_q = np.array([0.0, 0.0, 0.0, 1.0])  # No rotation

            r_prev = R.from_quat(q_prev)
            r_curr = r_prev * R.from_quat(delta_q)
            q_curr = r_curr.as_quat()
            q_curr = q_curr / np.linalg.norm(q_curr)  # renormalize
            quaternions_w[i] = q_curr

            # --- 2. Velocity and Position Update ---
            r_curr_matrix = r_curr.as_matrix()
            acc_world = r_curr_matrix @ acc_body - self.gravity # Adding gravity
            v_prev = velocities_w[i - 1]
            v_curr = v_prev + acc_world * dt
            velocities_w[i] = v_curr
            p_prev = positions_w[i - 1]
            p_curr = p_prev + v_prev * dt + 0.5 * acc_world * dt * dt
            positions_w[i] = p_curr

        # Final Output Formatting: Convert Quaternions to Euler angles (Roll, Pitch, Yaw)
        angles_rpy = R.from_quat(quaternions_w).as_euler('xyz', degrees=False) # Roll-Pitch-Yaw
        return np.hstack((angles_rpy, positions_w))
    


def visual_inertial_odometry(image_folder, imu_folder):
    dataLoader = DataLoader(image_folder=image_folder, imu_folder=imu_folder)
    visualOdom =VisualOdometry()
