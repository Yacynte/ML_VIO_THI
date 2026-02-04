import cv2 as cv
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import imu_processor_v2  # Import the imu module
import imu_to_cam  # Import the imu_to_cam module
import feature_matching  # Import the feature_mathing module


# -------------------- Config --------------------

alpha_t = 0.35 # Complementary filter weight for translation high -> trust VO more
alpha = 0.85  # Complementary filter weight for orientation high -> trust VO more
folder_number = 1  # Change this to process different folders


START_FRAME = 0  # Starting frame index
MAX_FRAMES_TO_PROCESS = -1  # Limit on number of frames to process

ESS_RANSAC_THRESH = 1.0
MIN_FEATURES_FOR_E = 8
MIN_INLIERS_TO_ACCEPT_POSE = 10

# Global pose storage (for writing to file)
LAST_POSE_R = None
LAST_POSE_t = None

global VIZUALISE_MATCHES
VIZUALISE_MATCHES = True
global STOP
STOP = False
image_timestamps = None
imu_poses = None  # To be loaded later
global image_files
global fused_poses, fused_rotations
fused_poses = []
fused_rotations = []
global min_indx

# -------------------- Load dataset Paths --------------------

# UrbanIng-V2X dataset paths
dataset_type = 'urbaning'
BASE_PATH = "/home/divan/ML_VIO_THI"
DATASET_PATH = os.path.join(BASE_PATH, f'datasets/UrbanIng-V2X/dataset/20241126_{folder_number:04d}_crossing2_00/vehicle2_front_left_camera')
imu_path = os.path.join(BASE_PATH, f'datasets/UrbanIng-V2X/dataset/20241126_{folder_number:04d}_crossing2_00/vehicle2_state')
calib_file = os.path.join(BASE_PATH, f'datasets/UrbanIng-V2X/dataset/20241126_{folder_number:04d}_crossing2_00/calibration.json')
timestamp_file = None
ground_truth_file = None
imu2velo_path = None
velo2cam_path = None
cam2cam_path = None
ess_tau = 15


# # Kitti dataset paths (Commented out)
# dataset_type = 'kitti'
# calib_file = None
# ground_truth_file = None
# # str2 = "09_26"
# str2 = "10_03"
# BASE_PATH = "/home/divan/ML_VIO_THI"
# imu_path = os.path.join(BASE_PATH, f'2011_{str2}_drive_{folder_number:04d}_sync/2011_{str2}/2011_{str2}_drive_{folder_number:04d}_sync/oxts')
# DATASET_PATH = os.path.join(BASE_PATH, f'2011_{str2}_drive_{folder_number:04d}_sync/2011_{str2}/2011_{str2}_drive_{folder_number:04d}_sync/image_00/data')
# timestamp_file = os.path.join(BASE_PATH, f'2011_{str2}_drive_{folder_number:04d}_sync/2011_{str2}/2011_{str2}_drive_{folder_number:04d}_sync/image_00/timestamps.txt')
# imu2velo_path = os.path.join(BASE_PATH, f'2011_{str2}_drive_{folder_number:04d}_sync/2011_{str2}/2011_{str2}_calib/2011_{str2}/calib_imu_to_velo.txt')
# velo2cam_path = os.path.join(BASE_PATH, f'2011_{str2}_drive_{folder_number:04d}_sync/2011_{str2}/2011_{str2}_calib/2011_{str2}/calib_velo_to_cam.txt')
# cam2cam_path = os.path.join(BASE_PATH, f'2011_{str2}_drive_{folder_number:04d}_sync/2011_{str2}/2011_{str2}_calib/2011_{str2}/calib_cam_to_cam.txt')
# ess_tau = 1

# KITTI-Mat dataset paths
# DATASET_PATH = f'/home/divan/ML_VIO_THI/Visual-Selective-VIO/data/sequences/{folder_number:02d}/image_2'
# imu_path = f'/home/divan/ML_VIO_THI/Visual-Selective-VIO/data/imus/{folder_number:02d}.mat'
# calib_file = f'/home/divan/ML_VIO_THI/Visual-Selective-VIO/data/sequences/{folder_number:02d}/calib.txt'
# timestamp_file = f'/home/divan/ML_VIO_THI/Visual-Selective-VIO/data/sequences/{folder_number:02d}/times.txt'
# ground_truth_file = f'/home/divan/ML_VIO_THI/Visual-Selective-VIO/data/poses/{folder_number:02d}.txt'


# -------------------- Load Transforms --------------------
transformer = imu_to_cam.TransformMatrix(dataset_type=dataset_type, camera='front_left_camera', vehicle='vehicle2')

# Load calibration
if dataset_type == 'urbaning':
    T_imu_cam = transformer.compute_T_cam_to_imu_urbaning(calib_file)
    K = transformer.intrinsics
    D = transformer.distortion
    print("\n--- Urbaning Calibration Loaded ---")
    print("K:\n", K)
    print("D:\n", D)
elif dataset_type == 'kitti':
    T_imu_cam = transformer.compute_T_imu_to_cam(imu2velo_path, velo2cam_path, cam2cam_path, cam_id=0)
    K = transformer.intrinsics
    D = transformer.distortion

# -------------------- Quaterion Interpolation --------------------
def quaternion_slerp(q1, q2, fraction):
    """Interpolates between two quaternions using Scipy Slerp."""
    from scipy.spatial.transform import Slerp
    key_rots = R.from_quat([q1, q2])
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    return slerp([fraction]).as_quat()[0]

# -------------------- Visual Odometry --------------------
def calculate_visual_odometry(p0_px, p1_px, K, dist_coeffs=None):
    """Compute Essential matrix, recover R,t."""
    if p0_px is None or p1_px is None:
        return None, None, None

    if len(p0_px) < MIN_FEATURES_FOR_E:
        print(f"Warning: need at least {MIN_FEATURES_FOR_E} matched points (got {len(p0_px)})")
        return None, None, None

    # Get focal length for threshold scaling
    f = (K[0,0] + K[1,1]) / 2.0
    
    # Use global ess_tau (set to 6.0 for urbaning)
    curr_tau = ess_tau

    if dist_coeffs is not None:
        # Undistort into NORMALIZED coordinates (P=None)
        # This is more robust to K inaccuracies
        p0 = cv.undistortPoints(p0_px.reshape(-1, 1, 2), K, dist_coeffs, P=None).reshape(-1, 2)
        p1 = cv.undistortPoints(p1_px.reshape(-1, 1, 2), K, dist_coeffs, P=None).reshape(-1, 2)
        # Essential Matrix units are now in normalized plane [x/z, y/z]
        E_K = np.eye(3)
        E_tau = curr_tau / f
    else:
        p0 = np.asarray(p0_px, dtype=np.float64).reshape(-1, 2)
        p1 = np.asarray(p1_px, dtype=np.float64).reshape(-1, 2)
        E_K = K
        E_tau = curr_tau

    # Essential matrix
    # Using RANSAC for Urbaning (USAC_MAGSAC can be too strict with normalized small thresholds)
    E, mask = cv.findEssentialMat(p0, p1, E_K, method=cv.RANSAC, prob=0.999, threshold=E_tau)

    if E is None:
        print("Error: findEssentialMat failed.")
        return None, None, None

    try:
        # recoverPose with the SAME points and E_K used for E
        retval, R, t, mask_pose = cv.recoverPose(E, p0, p1, E_K, mask=mask)
        num_inliers = int(retval)

    except:
        return None, None, None

    if R is None or t is None:
        return None, None, None

    if mask_pose is None:
        mask_pose = mask
    mask_pose = np.asarray(mask_pose).reshape(-1, 1)

    inlier_ratio = num_inliers / len(p0) * 100.0
    print(f"  > VO Inliers: {num_inliers}/{len(p0)} ({inlier_ratio:.2f}%)")

    if num_inliers < MIN_INLIERS_TO_ACCEPT_POSE:
        print("  > Warning: very low inlier count — pose unreliable.")

    # Normalize translation
    t = t / (np.linalg.norm(t) + 1e-12)

    return R, t, mask_pose


# -------------------- KLT + VO Tracking --------------------


def feature_detection(img_a_color, img_b_color, frame_index, method = 3):
    img_a_gray = cv.cvtColor(img_a_color, cv.COLOR_BGR2GRAY)
    img_b_gray = cv.cvtColor(img_b_color, cv.COLOR_BGR2GRAY)

    src_pts, dst_pts, kp1, kp2, matches, outlier_matches = None, None, None, None, None, None

    if method == 1:
        src_pts, dst_pts, success = feature_matching.shi_tomasi_klt_tracking(img_a_gray, img_b_gray, frame_index)
    elif method == 2:
        src_pts, dst_pts, success = feature_matching.orb_bf_matcher(img_a_color, img_b_color)
    elif method == 3:
        src_pts, dst_pts, kp1, kp2, matches, outlier_matches, success = feature_matching.sift_flann_matcher(img_a_gray, img_b_gray)

    return src_pts, dst_pts, kp1, kp2, matches, outlier_matches, success


def process_and_save_tracking(img_a_color, img_b_color, frame_index):
    global LAST_POSE_R, LAST_POSE_t

    if img_a_color is None or img_b_color is None:
        print(f"Skipping frame {frame_index}: failed to load images.")
        return False

    # tracking method
    # 1: Shi-Tomasi + KL-Optical Flow
    # 2: ORB + BFMatcher
    # 3: SIFT + FLANN
    method = 3 # Hardcoded or could be arg, but user wants method 3 logic
    
    # Feature detection + KLT tracking
    p0_tracked, p1_tracked, kp1, kp2, matches, outlier_matches, success = feature_detection(img_a_color, img_b_color, frame_index, method=method)

    print(f"\nFrame {frame_index} → {frame_index+1}: tracked {len(p0_tracked)} features.")

    # vizualise matches
    if VIZUALISE_MATCHES:
        vizualise_tracked(img_a_color, img_b_color, kp1 if method == 3 else p0_tracked, kp2 if method == 3 else p1_tracked, matches, None, frame_index, method)
    # Compute VO
    if len(p0_tracked) >= MIN_FEATURES_FOR_E and success:
        # Pass distortion coefficients if available (for Urbaning)
        dist_coeffs = D if dataset_type == 'urbaning' else None
        Rot, t, mask_vo = calculate_visual_odometry(p0_tracked, p1_tracked, K, dist_coeffs=dist_coeffs)
    else:
        Rot, t, mask_vo = None, None, None
        print("  > VO cannot run — not enough points.")

    
    # Apply Physical Scale from IMU
    t_imu = interpolated_imu_positions[frame_index + 1] - interpolated_imu_positions[frame_index] if interpolated_imu_positions is not None else None
    scale = np.linalg.norm(t_imu) if t_imu is not None else 1.0
    print(f"  > Interpolated IMU translation {frame_index} → {frame_index+1}:", interpolated_imu_positions[frame_index], interpolated_imu_positions[frame_index + 1])
    if t is not None:
    #     t = t * t_imu.reshape(3,1)
        t = t * scale
        print(f"  > Scale applied to VO translation: {scale:.4f}")
        # 4. Orientation Update
        # calculate_visual_odometry(p0, p1) uses recoverPose(E, p0, p1)
        # which returns Rot=R_01, t=t_01 (transformation from CURRENT(1) to PREVIOUS(0))
        # Thus t is the position of CURRENT camera center in PREVIOUS camera frame.
        R_rel = Rot.T # Transformation from PREVIOUS to CURRENT
        t_rel = -R_rel @ t.reshape(3, 1)

        # 5. Position Update
        # Rotate camera-frame increment by current world orientation to get world increment
        t_vo_world = LAST_POSE_R @ t_rel
        
        # Update orientation: R_w_curr = R_w_prev * R_prev_curr
        LAST_POSE_R = LAST_POSE_R @ R_rel
        
        # 2. IMU World increment (for fusion if needed)
        t_imu_world = t_imu.reshape(3, 1) if t_imu is not None else np.zeros((3, 1))
        
        # 3. Position Fusion (α_t trusts VO more if high)
        t_final = alpha_t * t_vo_world + (1 - alpha_t) * t_imu_world
        LAST_POSE_t += t_final

        # Optional: Orientation Fusion with IMU
        if alpha < 1.0:
            q_vo = R.from_matrix(LAST_POSE_R).as_quat()
            q_imu = R.from_euler('xyz', interpolated_imu_angles[frame_index + 1], degrees=False).as_quat()
            q_fused = quaternion_slerp(q_vo, q_imu, 1 - alpha)
            LAST_POSE_R = R.from_quat(q_fused).as_matrix()

        # ------------------------------------
        # print("  > R (Fused):")
        # print(np.round(LAST_POSE_R, 4))
        # print("  > t (Fused):", np.round(LAST_POSE_t.flatten(), 4))
        # print("  > t (imu):", np.round(interpolated_imu_positions[frame_index + 1].flatten(), 4))

        if fused_poses is not None:
            fused_poses.append(LAST_POSE_t.flatten().copy())
            fused_rotations.append(LAST_POSE_R.copy())

    return True


# -------------------- Vizualise Tracked Points --------------------
def vizualise_tracked(img_a_color, img_b_color, kp1, kp2, good_matches, outlier_matches, frame_index, method):
    # 4. Visualize and Save the Tracking Results
    
    # Combine the two images side-by-side for drawing
    h1, w1, _ = img_a_color.shape
    h2, w2, _ = img_b_color.shape
    combined_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    combined_img[:h1, :w1] = img_a_color
    combined_img[:h2, w1:w1+w2] = img_b_color

    if outlier_matches is not None:
        # Extract locations of outliers
        src_outs = np.float32([kp1[m.queryIdx].pt for m in outlier_matches]).reshape(-1, 2)
        dst_outs = np.float32([kp2[m.trainIdx].pt for m in outlier_matches]).reshape(-1, 2)
        for i in range(len(src_outs)):
            cv.circle(combined_img, (int(src_outs[i][0]), int(src_outs[i][1])), 3, (0, 0, 255), -1)
            cv.circle(combined_img, (int(dst_outs[i][0]) + w1, int(dst_outs[i][1])), 3, (0, 0, 255), -1)
            cv.line(combined_img, (int(src_outs[i][0]), int(src_outs[i][1])), (int(dst_outs[i][0]) + w1, int(dst_outs[i][1])), (0, 0, 255), 1)

    if good_matches is not None and method == 3:
        # Extract locations of good matches
        p0_tracked = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        p1_tracked = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    else:
        p0_tracked = kp1
        p1_tracked = kp2
    
    num_tracked = len(p0_tracked)
    # Draw lines connecting the matched points
    for i in range(num_tracked):
        # Draw on Frame A
        cv.circle(combined_img, (int(p0_tracked[i][0]), int(p0_tracked[i][1])), 3, (0, 255, 0), -1) # Green circle on Frame A
        
        # Draw on Frame B (remember to offset the X coordinate)
        x_frame2 = int(p1_tracked[i][0]) + w1
        y_frame2 = int(p1_tracked[i][1])
        cv.circle(combined_img, (x_frame2, y_frame2), 3, (0, 0, 255), -1) # Blue circle on Frame B
        
        # Draw line connecting them (Red line)
        cv.line(combined_img, 
                (int(p0_tracked[i][0]), int(p0_tracked[i][1])), 
                (x_frame2, y_frame2), 
                (255, 0, 0), 1)

    # Save the combined image
    save_filepath =  f'tracking_result_urbaning_turn_{frame_index:05d}.png'
    # cv.imwrite(save_filepath, combined_img)
    cv.imshow('Tracking Result', combined_img)
    # Capture the key press
    key = cv.waitKey(1) & 0xFF 

    if key == ord('q'):
        cv.destroyAllWindows()
        global STOP
        STOP = True
        
    elif key == ord('s'):
        cv.imwrite(save_filepath, combined_img)
        print(f"  > Saved tracking result to {save_filepath}")


# -------------------- Main Loop with Pose Saving --------------------
def vins_visual_tracking_demo():
    global image_timestamps, imu_poses, gps_coords
    
    # Load image timestamps first to determine time range
    all_image_timestamps = load_timestamp(dataset_type=dataset_type)
    image_timestamps = all_image_timestamps[START_FRAME:MAX_FRAMES_TO_PROCESS if MAX_FRAMES_TO_PROCESS is not None else len(all_image_timestamps)].copy()

    if image_timestamps is None or len(image_timestamps) == 0:
        print("Failed to load image timestamps.")
        return

    print("Computing IMU poses using imu_processor_v2...")
    
    # Updated API returns tuple: (imu_poses, gps_coords)
    # Using timestamps for robust slicing across different data frequencies (e.g. Urbaning)
    imu_poses, gps_coords_ = imu_processor_v2.get_processed_imu_data(dataset_type=dataset_type, 
                                                                    imu_folder=imu_path, 
                                                                    timestamp_file=timestamp_file,
                                                                    start_time=image_timestamps[0],
                                                                    end_time=image_timestamps[-1]
                                                                )

    if imu_poses is None:
        print("Failed to load IMU data.")
        return
    # Add initial pose
    global fused_poses, fused_rotations
    fused_poses = []
    fused_rotations = []
    # Interpolate IMU poses to match image timestamps
    global interpolated_imu_positions, interpolated_imu_angles
    if imu_poses is not None and image_timestamps is not None:
        imu_timestamps = imu_poses[:, -1]  # Last column is timestamp
        
        # Now that imu_poses is pre-sliced, min_indx should be close to 0
        min_indx = np.abs(imu_timestamps - image_timestamps[0]).argmin()
        max_indx = np.abs(imu_timestamps - image_timestamps[-1]).argmin()
        
        # Match GPS coordinates
        gps_coords = gps_coords_.copy() if gps_coords_ is not None else None 
        
        # Interpolate IMU Positions (X, Y, Z) to match exact image timestamps
        interpolated_imu_positions = np.column_stack([
            np.interp(image_timestamps, imu_timestamps, imu_poses[:, 3]),
            np.interp(image_timestamps, imu_timestamps, imu_poses[:, 4]),
            np.interp(image_timestamps, imu_timestamps, imu_poses[:, 5]),
        ])

        # Interpolate IMU Rotations (Roll, Pitch, Yaw)
        interpolated_imu_angles = np.column_stack([
            np.interp(image_timestamps, imu_timestamps, imu_poses[:, 0]),
            np.interp(image_timestamps, imu_timestamps, imu_poses[:, 1]),
            np.interp(image_timestamps, imu_timestamps, imu_poses[:, 2]),
        ])
        
        int_psose = []
        int_rot_vecs = []
        for i in range(len(image_timestamps)):
            int_rot_vec, int_pose = imu_to_cam.imu_pose_to_camera_pose(interpolated_imu_angles[i], interpolated_imu_positions[i], T_imu_cam)
            int_psose.append(int_pose)
            int_rot_vecs.append(int_rot_vec)
        
        # Add initial camera pose from IMU/GPS to the fused trajectory
        fused_poses.append(int_psose[0].copy())
        fused_rotations.append(R.from_euler('xyz', int_rot_vecs[0], degrees=False).as_matrix())
        
        # Overwrite with camera-frame interpolated poses if needed for VO logic
        interpolated_imu_positions = np.array(int_psose)
        interpolated_imu_angles = np.array(int_rot_vecs)
        # plot_imu_gps_3d(interpolated_imu_positions_, interpolated_imu_positions)
    else:
        interpolated_imu_positions = None

    global image_files
    image_files = sorted(glob.glob(os.path.join(DATASET_PATH, "*.[pj][np][ge]*")))  # matches png, jpg, jpeg)
    if not image_files:
        print("ERROR: No images found.")
        return

    image_files = image_files[START_FRAME:MAX_FRAMES_TO_PROCESS if MAX_FRAMES_TO_PROCESS is not None else len(image_files)].copy()  # Limit to max frames to process

    # Initial Orientation of Camera in World: R_wc = R_wi * R_ic
    global LAST_POSE_R, LAST_POSE_t
    
    # Initialize from the FIRST interpolated pose
    LAST_POSE_R = fused_rotations[0].copy()
    LAST_POSE_t = fused_poses[0].copy().reshape(3, 1)
    
    # Ensure they are stored for the main loop
    assert LAST_POSE_R is not None and LAST_POSE_t is not None, "Initial pose not set correctly."
    # R_ci = T_imu_cam[:3, :3]
    # R_ic = R_ci.T
    # R_0 = R_wi @ R_ic

    # global LAST_POSE_R, LAST_POSE_t
    # p0 = np.eye(4)
    # p0[:3, :3] = R_0
    # p0[:3, 3] = gps_coords[0] if gps_coords is not None else np.zeros((3,))
    # p0_c = p0 @ np.linalg.inv(T_imu_cam)
    # LAST_POSE_t = p0_c[:3, 3].reshape(3, 1)

    # r_vec0 = interpolated_imu_angles_[min_indx] if interpolated_imu_angles is not None else np.zeros((3,))
    # LAST_POSE_R = R_0.copy()
    # LAST_POSE_R = R_wi.copy()
    # print("Initial camera position (from GPS): ", LAST_POSE_t.flatten())
    # print("Initial camera position (from IMU): ", interpolated_imu_positions[0].flatten())
    
    
    # if fused_poses is not None:
    #     fused_poses.append(LAST_POSE_t.flatten().copy())
    #     fused_rotations.append(LAST_POSE_R.copy())
    
    for i in range(len(image_files) - 1):
        imgA = cv.imread(image_files[i])
        imgB = cv.imread(image_files[i + 1])

        success = process_and_save_tracking(imgA, imgB, i)
        if STOP:
            break

    # Convert fused poses to imu frame for plotting
    fused_poses_imu = []
    
    # Transform path from Camera World frame to IMU World frame for comparison
    for i in range(len(fused_poses)):
        f_rot = fused_rotations[i]
        f_t = fused_poses[i]
        _, imu_t_w = imu_to_cam.camera_pose_to_imu_pose(f_rot, f_t, T_imu_cam)
        fused_poses_imu.append(imu_t_w)
    
    final_trajectory = np.array(fused_poses_imu)
    camera_poses = np.array(fused_poses)
    
    if len(final_trajectory) > 0:
        print("Plotting VIO trajectory (aligned to IMU frame) vs Ground Truth...")
        # np.save("cam_poses.npy", camera_poses)
        # np.save("imu_cam_coord.npy", interpolated_imu_positions)
        print("length of final trajectory:", len(final_trajectory))
        print("length of ground truth:", len(gps_coords))
        # plot_imu_gps_3d(final_trajectory, gps_pose=gps_coords)
        plot_imu_gps_2d(final_trajectory, gps_pose=gps_coords)
        # plot_result(final_trajectory, gps_pose=gps_coords)
        
    print("\n--- Done ---")
    # print("Pose file saved at:", POSE_FILE)


def load_timestamp(dataset_type = 'kitti') -> np.ndarray:
        """Loads timestamp data from image files."""
        timestamps = []
        if dataset_type == 'urbaning':
            for filename in sorted(os.listdir(DATASET_PATH)):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    timestamp_str = filename.replace('.png', '').replace('.jpg', '')
                    pd_timestamp = pd.to_datetime(int(timestamp_str), unit="ms")
                    timestamp_sec = pd_timestamp.timestamp()
                    timestamps.append(timestamp_sec)
            timestamps = np.array(timestamps, dtype=np.float64)
            return timestamps
        
        if dataset_type == 'kitti_mat':
            # Support for .mat or existing files
            pass # Simplified for clarity, assuming generic text file path for now
        
        # Kitti / Default Text File
        if timestamp_file and os.path.exists(timestamp_file):
            with open(timestamp_file, 'r') as f:
                for line in f:
                    timestamps_str = line.strip()
                    try:
                        # Try parsing as datetime string
                        pd_timestamp = pd.to_datetime(timestamps_str)
                        timestamp_sec = pd_timestamp.timestamp()
                    except:
                        # Fallback to float
                        timestamp_sec = float(timestamps_str)
                    timestamps.append(timestamp_sec)
            timestamps = np.array(timestamps, dtype=np.float64)
            return timestamps
        return np.array([])

def plot_imu_gps_3d(pose_ins: np.ndarray, gps_pose: np.ndarray = None):
    """Plot 3D trajectory: IMU/VIO (Blue) vs GPS (Red)."""
    dist = np.linalg.norm(pose_ins[-1] - pose_ins[0])
    print(f"Total Distance Travelled (VIO): {dist:.2f} meters")
    
    # Defer imports
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    try:
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot(pose_ins[:, 0], pose_ins[:, 1], pose_ins[:, 2], label='VIO Trajectory', color='blue', linewidth=2)
        
        if gps_pose is not None:
            # Downsample GPS for plotting speed if needed, or plot all
            ax.plot(gps_pose[:, 0], gps_pose[:, 1], gps_pose[:, 2], label='Ground Truth (GPS)', color='red', linestyle='--', linewidth=2)
            
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D VIO Trajectory vs Ground Truth')
        ax.legend()
        
        # Equal axis aspect ratio hack
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
        plt.savefig("vio_trajectory_kitti_2_3d.png")
        plt.show()
    except Exception as e:
        print(f"Plotting failed: {e}")

def plot_imu_gps_2d(pose_ins: np.ndarray, gps_pose: np.ndarray = None):
    dist = pose_ins[-1, :] - pose_ins[0, :]
    print(f"Total Distance Travelled (IMU): {dist} meters")
    plt.figure()
    plt.plot(pose_ins[:, 0], pose_ins[:, 1], label='VIO Trajectory', color='blue', linewidth=2)
    if gps_pose is not None:
        plt.plot(gps_pose[:, 0], gps_pose[:, 1], label='Ground Truth', linestyle='--', color='red', linewidth=2)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('2D VIO Trajectory VS Ground truth, urbaning data path one')
    plt.legend()
    plt.axis('equal')
    plt.grid()
    plt.savefig("vio_trajectory_kitti_klt.png")
    plt.show()

def plot_result(pose_ins: np.ndarray, gps_pose: np.ndarray = None):
    error = np.linalg.norm(gps_pose[:, :2] - pose_ins[:, :2], axis=1)
    # dist = gps_pose[:, :2] - gps_pose[0, :2]
    # dist = np.linalg.norm(dist, axis=1)
    dist = [0]
    for i in range(1, len(gps_pose)):
        dx = gps_pose[i, :2] - gps_pose[i-1, :2]
        dist.append(np.linalg.norm(dx) + dist[i-1])
    dist = np.array(dist)
    fig, axs = plt.subplots(1, 2, figsize=(12, 7))
    ax1 = axs[0]
    ax1.plot(pose_ins[:, 0], pose_ins[:, 1], label='VIO Trajectory', color='blue', linewidth=2)
    if gps_pose is not None:
        ax1.plot(gps_pose[:, 0], gps_pose[:, 1], label='Ground Truth', linestyle='--', color='red', linewidth=2)
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('2D VIO Trajectory VS Ground truth')
    ax1.legend()
    ax1.axis('equal')
    ax1.grid()

    ax2 = axs[1]
    ax2.plot(dist, error, label='Error', color='green', linewidth=2)
    ax2.set_xlabel('Distance travelled (m)')
    ax2.set_ylabel('Error (m)')
    ax2.set_title('Error over Time')
    ax2.legend()
    ax2.grid()

    plt.title("urbaning data path 1")
    plt.tight_layout()
    plt.savefig("vio_kitti_sift_w_error.png")
    plt.show()

if __name__ == "__main__":
    # START_FRAME = 0
    # MAX_FRAMES_TO_PROCESS = 50
    vins_visual_tracking_demo()
