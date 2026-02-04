import cv2 as cv
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import imu  # Import the imu module
import imu_to_cam  # Import the imu_to_cam module
import feature_matching  # Import the feature_mathing module
# -------------------- Config --------------------
FOCAL_LENGTH = 718.856
PP = (607.1928, 185.2157)  # (cx, cy)

K = np.array([
    [FOCAL_LENGTH, 0, PP[0]],
    [0, FOCAL_LENGTH, PP[1]],
    [0, 0, 1]
], dtype=np.float64)

folder_number = 1  # Change this to process different folders

# DATASET_PATH = r'C:/Users/Asus/Downloads/VS Studio/New folder/VINS MONO/dataset/image_0'
OUTPUT_FOLDER = r'C:/Users/Asus/Downloads/VS Studio/New folder/VINS MONO/klt_tracking_results_vo'

# DATASET_PATH = f'/home/divan/ML_VIO_THI/2011_09_26_drive_00{folder_number:02d}_sync/2011_09_26/2011_09_26_drive_00{folder_number:02d}_sync/image_00/data'
POSE_FILE = os.path.join(OUTPUT_FOLDER, "vo_poses.txt")

MAX_FRAMES_TO_PROCESS = 200

ESS_RANSAC_THRESH = 1.0
MIN_FEATURES_FOR_E = 8
MIN_INLIERS_TO_ACCEPT_POSE = 10
# ------------------------------------------------

# Global pose storage (for writing to file)
LAST_POSE_R = None
LAST_POSE_t = None


image_timestamps = None
imu_poses = None  # To be loaded later
global image_files
global fused_poses
fused_poses = []
fused_rotations = []
dataset_type = 'kitti'  # 'kitti', 'kitti_mat', 'urbaning'

# Kitti dataset paths
calib_file = None
ground_truth_file = None
imu_path = f'2011_09_26_drive_{folder_number:04d}_sync/2011_09_26/2011_09_26_drive_{folder_number:04d}_sync/oxts'
DATASET_PATH = f'/home/divan/ML_VIO_THI/2011_09_26_drive_{folder_number:04d}_sync/2011_09_26/2011_09_26_drive_{folder_number:04d}_sync/image_00/data'
timestamp_file = f'/home/divan/ML_VIO_THI/2011_09_26_drive_{folder_number:04d}_sync/2011_09_26/2011_09_26_drive_{folder_number:04d}_sync/image_00/timestamps.txt'
imu2velo_path = f'/home/divan/ML_VIO_THI/2011_09_26_drive_{folder_number:04d}_sync/2011_09_26/2011_09_26_calib/2011_09_26/calib_imu_to_velo.txt'
velo2cam_path = f'/home/divan/ML_VIO_THI/2011_09_26_drive_{folder_number:04d}_sync/2011_09_26/2011_09_26_calib/2011_09_26/calib_velo_to_cam.txt'
cam2cam_path = f'/home/divan/ML_VIO_THI/2011_09_26_drive_{folder_number:04d}_sync/2011_09_26/2011_09_26_calib/2011_09_26/calib_cam_to_cam.txt'

# # UrbanIng-V2X dataset paths
# DATASET_PATH = f'datasets/UrbanIng-V2X/dataset/20241126_{folder_number:04d}_crossing2_00/vehicle2_front_left_camera'
# imu_path = f'datasets/UrbanIng-V2X/dataset/20241126_{folder_number:04d}_crossing2_00/vehicle2_state'
# calib_file = f'datasets/UrbanIng-V2X/dataset/20241126_{folder_number:04d}_crossing2_00/calibration.json'
# timestamp_file = None
# ground_truth_file = None
# imu2velo_path = None
# velo2cam_path = None
# cam2cam_path = None

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
elif dataset_type == 'kitti':
    T_imu_cam = transformer.compute_T_imu_to_cam(imu2velo_path, velo2cam_path, cam2cam_path, cam_id=0)
    K = transformer.intrinsics
    D = transformer.distortion
# -------------------- Visual Odometry --------------------
def calculate_visual_odometry(p0_px, p1_px, K):
    """Compute Essential matrix, recover R,t."""
    if p0_px is None or p1_px is None:
        return None, None, None

    if len(p0_px) < MIN_FEATURES_FOR_E:
        print(f"Warning: need at least {MIN_FEATURES_FOR_E} matched points (got {len(p0_px)})")
        return None, None, None

    p0 = np.asarray(p0_px, dtype=np.float64).reshape(-1, 2)
    p1 = np.asarray(p1_px, dtype=np.float64).reshape(-1, 2)

    # Essential matrix (pixel coords + K)
    E, mask = cv.findEssentialMat(p0, p1, K, method=cv.RANSAC, prob=0.999, threshold=3.0)

    if E is None:
        print("Error: findEssentialMat failed.")
        return None, None, None

    # recoverPose — support both OpenCV signatures
    try:
        retval, R, t, mask_pose = cv.recoverPose(E, p0, p1, K, mask=mask)
        num_inliers = int(retval)
    except:
        try:
            R, t, mask_pose = cv.recoverPose(E, p0, p1, K, mask=mask)
            num_inliers = int(np.sum(mask_pose))
        except Exception as e:
            print("recoverPose failed:", e)
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

    # Normalize translation (monocular scale unknown)
    t = t / (np.linalg.norm(t) + 1e-12)

    return R, t, mask_pose


# -------------------- KLT + VO Tracking --------------------


def feature_detection(img_a_color, img_b_color, frame_index, method = 1):
    img_a_gray = cv.cvtColor(img_a_color, cv.COLOR_BGR2GRAY)
    img_b_gray = cv.cvtColor(img_b_color, cv.COLOR_BGR2GRAY)

    src_pts, dst_pts, kp1, kp2, matches = None, None, None, None, None

    if method == 1:
        src_pts, dst_pts, success = feature_matching.shi_tomasi_klt_tracking(img_a_gray, img_b_gray, frame_index)
    elif method == 2:
        src_pts, dst_pts, success = feature_matching.orb_bf_matcher(img_a_color, img_b_color)
    elif method == 3:
        src_pts, dst_pts, kp1, kp2, matches, success = feature_matching.sift_flann_matcher(img_a_gray, img_b_gray)

    return src_pts, dst_pts, kp1, kp2, matches, success


def process_and_save_tracking(img_a_color, img_b_color, output_path, frame_index):
    global LAST_POSE_R, LAST_POSE_t

    if img_a_color is None or img_b_color is None:
        print(f"Skipping frame {frame_index}: failed to load images.")
        return False

    # tracking method
    # 1: Shi-Tomasi + KL-Optical Flow
    # 2: ORB + BFMatcher
    # 3: SIFT + FLANN
    method = 3
    # Feature detection + KLT tracking
    p0_tracked, p1_tracked, kp1, kp2, matches, success = feature_detection(img_a_color, img_b_color, frame_index, method=method)

    print(f"\nFrame {frame_index} → {frame_index+1}: tracked {len(p0_tracked)} features.")

    # Compute VO
    if len(p0_tracked) >= MIN_FEATURES_FOR_E and success:
        if method in [1, 2]:
            R, t, mask_vo = calculate_visual_odometry(p0_tracked, p1_tracked, K)
        elif method == 3:
            # R, t, mask_vo = feature_matching.recover_pose_robust(img_a_color, img_b_color, kp1=kp1, kp2=kp2, matches=matches, K=K, dist_coeffs=D)
            R, t, mask_vo = feature_matching.get_pose(p0_tracked, p1_tracked, K, dist_coeffs=D if dataset_type == 'urbaning' else None)
    else:
        R, t, mask_vo = None, None, None
        print("  > VO cannot run — not enough points.")

    # Store pose globally (for writing to file later)
    t_imu = interpolated_imu_positions[frame_index + 1] - interpolated_imu_positions[frame_index] if interpolated_imu_positions is not None else None
    t_imu = None
    t *= t_imu.reshape(3, 1) if t_imu is not None else 1.0

    if R is not None and t is not None:
        LAST_POSE_R = LAST_POSE_R @ R if LAST_POSE_R is not None else R.copy()
        LAST_POSE_t = LAST_POSE_t + (LAST_POSE_R @ t) if LAST_POSE_t is not None else (R @ t).copy()

        print("  > R:")
        print(np.round(LAST_POSE_R, 4))
        print("  > t (dir.):", np.round(LAST_POSE_t.flatten(), 4))

        if fused_poses is not None:
            fused_poses.append(np.round(LAST_POSE_t.flatten(), 4))
            fused_rotations.append(LAST_POSE_R.copy())

    # # ---- Visualization section ----
    # h1, w1 = img_a_color.shape[:2]
    # h2, w2 = img_b_color.shape[:2]
    # H = max(h1, h2)
    # W = w1 + w2

    # combined = np.zeros((H, W, 3), dtype=np.uint8)
    # combined[:h1, :w1] = img_a_color
    # combined[:h2, w1:w1 + w2] = img_b_color

    # if mask_vo is None:
    #     mask_vis = np.ones(len(p0_tracked), dtype=bool)
    # else:
    #     mask_vo = mask_vo.reshape(-1)
    #     mask_vis = mask_vo == 1 if len(mask_vo) == len(p0_tracked) else np.ones(len(p0_tracked), bool)

    # for i in range(len(p0_tracked)):
    #     pt0 = (int(p0_tracked[i, 0]), int(p0_tracked[i, 1]))
    #     pt1 = (int(p1_tracked[i, 0] + w1), int(p1_tracked[i, 1]))

    #     if mask_vis[i]:
    #         cv.circle(combined, pt0, 3, (0, 255, 0), -1)
    #         cv.circle(combined, pt1, 3, (255, 0, 0), -1)
    #         cv.line(combined, pt0, pt1, (0, 0, 255), 1)
    #     else:
    #         cv.circle(combined, pt0, 2, (0, 150, 0), -1)
    #         cv.circle(combined, pt1, 2, (150, 0, 0), -1)

    # # Save visualization
    # # save_path = os.path.join(output_path, f"tracking_result_{frame_index:05d}.png")
    # # cv.imwrite(save_path, combined)
    # cv.imshow("KLT Tracking and VO", combined)
    # cv.waitKey(1)

    return True


# -------------------- Main Loop with Pose Saving --------------------
def vins_visual_tracking_demo():

    global imu_poses, gps_coords
    # imu_poses = np.load('imu_positions.npy')  # Load precomputed IMU positions
    imu_poses, gps_coords = imu.load_imu_positions(imu_path, dataset_type=dataset_type, timestamp_file=timestamp_file, ground_truth_file=ground_truth_file)  # Load IMU positions

    # plot_imu_gps_2d(imu_poses[:, 3:6], gps_pose=gps_coords)
    global image_timestamps
    image_timestamps = load_timestamp(dataset_type=dataset_type)  # Load image timestamps

    # Interpolate IMU poses to match image timestamps
    global interpolated_imu_positions
    if imu_poses is not None and image_timestamps is not None:
        # Assuming imu_poses is a list of [x, y, z, timestamp] arrays
        imu_timestamps = imu_poses[:, -1]  # Last column is timestamp
        # Interpolate to get IMU positions at image timestamps
        # interpolated_imu_positions = np.interp(image_timestamps, imu_timestamps, imu_poses[:, :3], left=0, right=0)
        interpolated_imu_positions_ = np.column_stack([
            np.interp(image_timestamps, imu_timestamps, imu_poses[:, 3], left=imu_poses[0, 3], right=imu_poses[-1, 3]),
            np.interp(image_timestamps, imu_timestamps, imu_poses[:, 4], left=imu_poses[0, 4], right=imu_poses[-1, 4]),
            np.interp(image_timestamps, imu_timestamps, imu_poses[:, 5], left=imu_poses[0, 5], right=imu_poses[-1, 5]),
        ])
        interpolated_imu_angles = np.column_stack([
            np.interp(image_timestamps, imu_timestamps, imu_poses[:, 0], left=imu_poses[0, 0], right=imu_poses[-1, 0]),
            np.interp(image_timestamps, imu_timestamps, imu_poses[:, 1], left=imu_poses[0, 1], right=imu_poses[-1, 1]),
            np.interp(image_timestamps, imu_timestamps, imu_poses[:, 2], left=imu_poses[0, 2], right=imu_poses[-1, 2]),
        ])
        int_psose = []
        for i in range(len(interpolated_imu_angles)):
            int_rot_vec, int_pose = imu_to_cam.imu_pose_to_camera_pose(interpolated_imu_angles[i], interpolated_imu_positions_[i], T_imu_cam)
            int_psose.append(int_pose)
        interpolated_imu_positions = np.array(int_psose)
        
        # interpolated_imu_positions = (R_cam_imu @ interpolated_imu_positions.T).T
    else:
        interpolated_imu_positions = None

    # # Create output folder
    # if not os.path.exists(OUTPUT_FOLDER):
    #     os.makedirs(OUTPUT_FOLDER)

    # # Create pose log file
    # with open(POSE_FILE, "w") as f:
    #     f.write("# VO poses (R and t) per frame pair\n")
    #     f.write("# format:\n")
    #     f.write("# frame_i frame_i+1\n")
    #     f.write("# R (3x3)\n")
    #     f.write("# t (3x1 unit direction)\n\n")

    global image_files
    image_files = sorted(glob.glob(os.path.join(DATASET_PATH, "*.[pj][np][ge]*")))  # matches png, jpg, jpeg)
    if not image_files:
        print("ERROR: No images found.")
        return

    images_to_process = image_files[:MAX_FRAMES_TO_PROCESS]

    global LAST_POSE_R, LAST_POSE_t
    LAST_POSE_t = interpolated_imu_positions[0].reshape(3, 1) if interpolated_imu_positions is not None else np.zeros((3, 1))
    # LAST_POSE_t = np.zeros((3, 1))
    LAST_POSE_R = np.eye(3)
    for i in range(len(images_to_process) - 1):
        imgA = cv.imread(images_to_process[i])
        imgB = cv.imread(images_to_process[i + 1])

        success = process_and_save_tracking(imgA, imgB, OUTPUT_FOLDER, i)

        # # Write pose if available
        # if success and LAST_POSE_R is not None:
        #     with open(POSE_FILE, "a") as f:
        #         f.write(f"{i} {i+1}\n")
        #         for row in LAST_POSE_R:
        #             f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
        #         f.write(" ".join(f"{v:.6f}" for v in LAST_POSE_t.flatten()) + "\n\n")

    # Convert fused poses to imu frame for plotting
    global fused_poses
    fused_poses_imu = []
    for i in range(len(fused_poses)):
        fused_rot = fused_rotations[i]
        fused_t = fused_poses[i]
        r_imu, imu_t = imu_to_cam.camera_pose_to_imu_pose(fused_rot, fused_t, T_imu_cam)
        fused_poses_imu.append(imu_t)
    fused_poses = np.array(fused_poses_imu)
    if fused_poses is not None and len(fused_poses) > 0:
        # plot_imu_gps_2d(np.array(fused_poses), gps_pose=gps_coords)
        plot_imu_gps_3d(np.array(fused_poses), gps_pose=gps_coords)
    # Write all fused poses to file
    print("\n--- Done ---")
    print("Pose file saved at:", POSE_FILE)


def load_timestamp(dataset_type = 'kitti') -> np.ndarray:
        """Loads timestamp data from all OXTS files in the specified folder."""
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
            with open(timestamp_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:  # skip empty lines
                        timestamps.append(float(line))
            timestamps = np.array(timestamps, dtype=np.float64)
            return timestamps
        
        with open(timestamp_file, 'r') as f:
            for line in f:
                timestamps_str = line.strip()
                pd_timestamp = pd.to_datetime(timestamps_str)
                timestamp_sec = pd_timestamp.timestamp()
                timestamps.append(timestamp_sec)
        timestamps = np.array(timestamps, dtype=np.float64)
        return timestamps

def plot_imu_gps_2d(pose_ins: np.ndarray, gps_pose: np.ndarray = None):
    """Plot 2D trajectory from IMU poses and GPS poses."""
    # pose_ins = (np.linalg.inv(R_cam_imu) @ pose_ins[:, :3].T).T  # Transform camera to imu frame
    dist = pose_ins[-1] - pose_ins[0]
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

# -------------------- Run Script --------------------
if __name__ == "__main__":
    vins_visual_tracking_demo()
