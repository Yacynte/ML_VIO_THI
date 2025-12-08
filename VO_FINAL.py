import cv2 as cv
import numpy as np
import os
import glob

# -------------------- Config --------------------
FOCAL_LENGTH = 718.856
PP = (607.1928, 185.2157)  # (cx, cy)

K = np.array([
    [FOCAL_LENGTH, 0, PP[0]],
    [0, FOCAL_LENGTH, PP[1]],
    [0, 0, 1]
], dtype=np.float64)

DATASET_PATH = r'C:/Users/Asus/Downloads/VS Studio/New folder/VINS MONO/dataset/image_0'
OUTPUT_FOLDER = r'C:/Users/Asus/Downloads/VS Studio/New folder/VINS MONO/klt_tracking_results_vo'
POSE_FILE = os.path.join(OUTPUT_FOLDER, "vo_poses.txt")

MAX_FRAMES_TO_PROCESS = 200

ESS_RANSAC_THRESH = 1.0
MIN_FEATURES_FOR_E = 8
MIN_INLIERS_TO_ACCEPT_POSE = 10
# ------------------------------------------------

# Global pose storage (for writing to file)
LAST_POSE_R = None
LAST_POSE_t = None


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
    E, mask = cv.findEssentialMat(
        p0, p1, K,
        method=cv.RANSAC, prob=0.999,
        threshold=ESS_RANSAC_THRESH
    )

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
def process_and_save_tracking(img_a_color, img_b_color, output_path, frame_index):
    global LAST_POSE_R, LAST_POSE_t

    if img_a_color is None or img_b_color is None:
        print(f"Skipping frame {frame_index}: failed to load images.")
        return False

    img_a_gray = cv.cvtColor(img_a_color, cv.COLOR_BGR2GRAY)
    img_b_gray = cv.cvtColor(img_b_color, cv.COLOR_BGR2GRAY)

    # Shi-Tomasi features
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    p0 = cv.goodFeaturesToTrack(img_a_gray, mask=None, **feature_params)

    if p0 is None:
        print(f"No features in frame {frame_index}.")
        return False

    p0 = p0.reshape(-1, 2).astype(np.float32)

    # Optical flow tracking
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))
    p1, st, err = cv.calcOpticalFlowPyrLK(img_a_gray, img_b_gray, p0, None, **lk_params)

    if p1 is None:
        print(f"KLT failed for frame {frame_index}.")
        return False

    st = st.reshape(-1)
    err = err.reshape(-1) if err is not None else np.zeros_like(st)

    # Basic valid mask
    valid = (st == 1)
    if valid.sum() == 0:
        print(f"No valid tracked points at frame {frame_index}.")
        return False

    # Remove bad error points (2*std)
    e_th = np.mean(err[valid]) + 2 * np.std(err[valid])
    valid = valid & (err <= max(e_th, 12.0))

    p0_tracked = p0[valid]
    p1_tracked = p1.reshape(-1, 2)[valid]

    print(f"\nFrame {frame_index} → {frame_index+1}: tracked {len(p0_tracked)} features.")

    # Compute VO
    if len(p0_tracked) >= MIN_FEATURES_FOR_E:
        R, t, mask_vo = calculate_visual_odometry(p0_tracked, p1_tracked, K)
    else:
        R, t, mask_vo = None, None, None
        print("  > VO cannot run — not enough points.")

    # Store pose globally (for writing to file later)
    if R is not None and t is not None:
        LAST_POSE_R = R.copy()
        LAST_POSE_t = t.copy()

        print("  > R:")
        print(np.round(R, 4))
        print("  > t (dir.):", np.round(t.flatten(), 4))

    # ---- Visualization section ----
    h1, w1 = img_a_color.shape[:2]
    h2, w2 = img_b_color.shape[:2]
    H = max(h1, h2)
    W = w1 + w2

    combined = np.zeros((H, W, 3), dtype=np.uint8)
    combined[:h1, :w1] = img_a_color
    combined[:h2, w1:w1 + w2] = img_b_color

    if mask_vo is None:
        mask_vis = np.ones(len(p0_tracked), dtype=bool)
    else:
        mask_vo = mask_vo.reshape(-1)
        mask_vis = mask_vo == 1 if len(mask_vo) == len(p0_tracked) else np.ones(len(p0_tracked), bool)

    for i in range(len(p0_tracked)):
        pt0 = (int(p0_tracked[i, 0]), int(p0_tracked[i, 1]))
        pt1 = (int(p1_tracked[i, 0] + w1), int(p1_tracked[i, 1]))

        if mask_vis[i]:
            cv.circle(combined, pt0, 3, (0, 255, 0), -1)
            cv.circle(combined, pt1, 3, (255, 0, 0), -1)
            cv.line(combined, pt0, pt1, (0, 0, 255), 1)
        else:
            cv.circle(combined, pt0, 2, (0, 150, 0), -1)
            cv.circle(combined, pt1, 2, (150, 0, 0), -1)

    # Save visualization
    save_path = os.path.join(output_path, f"tracking_result_{frame_index:05d}.png")
    cv.imwrite(save_path, combined)

    return True


# -------------------- Main Loop with Pose Saving --------------------
def vins_visual_tracking_demo():

    # Create output folder
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Create pose log file
    with open(POSE_FILE, "w") as f:
        f.write("# VO poses (R and t) per frame pair\n")
        f.write("# format:\n")
        f.write("# frame_i frame_i+1\n")
        f.write("# R (3x3)\n")
        f.write("# t (3x1 unit direction)\n\n")

    image_files = sorted(glob.glob(os.path.join(DATASET_PATH, "*.png")))
    if not image_files:
        print("ERROR: No images found.")
        return

    images_to_process = image_files[:MAX_FRAMES_TO_PROCESS]

    global LAST_POSE_R, LAST_POSE_t

    for i in range(len(images_to_process) - 1):
        imgA = cv.imread(images_to_process[i])
        imgB = cv.imread(images_to_process[i + 1])

        success = process_and_save_tracking(imgA, imgB, OUTPUT_FOLDER, i)

        # Write pose if available
        if success and LAST_POSE_R is not None:
            with open(POSE_FILE, "a") as f:
                f.write(f"{i} {i+1}\n")
                for row in LAST_POSE_R:
                    f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
                f.write(" ".join(f"{v:.6f}" for v in LAST_POSE_t.flatten()) + "\n\n")

    print("\n--- Done ---")
    print("Pose file saved at:", POSE_FILE)


# -------------------- Run Script --------------------
if __name__ == "__main__":
    vins_visual_tracking_demo()
