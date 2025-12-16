import cv2 as cv
import numpy as np
import os
import glob

# ==================== CAMERA INTRINSICS (KITTI) ====================
FOCAL_LENGTH = 718.856
PP = (607.1928, 185.2157)  # (cx, cy)

K = np.array([
    [FOCAL_LENGTH, 0, PP[0]],
    [0, FOCAL_LENGTH, PP[1]],
    [0, 0, 1]
], dtype=np.float64)

# ==================== PATHS ====================
DATASET_PATH = r'C:/Users/Asus/Downloads/VS Studio/New folder/VINS MONO/dataset/07/image_0'
TIMESTAMP_FILE = r'C:/Users/Asus/Downloads/VS Studio/New folder/VINS MONO/dataset/07/times.txt'
OUTPUT_FOLDER = r'D:/VO_time/vo'
POSE_FILE = os.path.join(OUTPUT_FOLDER, "vo_poses_with_timestamps.txt")

MAX_FRAMES = 200

ESS_RANSAC_THRESH = 1.0
MIN_FEATURES = 8
MIN_INLIERS = 10
# ================================================================

LAST_POSE_R = None
LAST_POSE_t = None


# ==================== KITTI TIMESTAMP LOADER ====================
def load_kitti_timestamps(path):
    timestamps = []
    with open(path, 'r') as f:
        for line in f:
            timestamps.append(float(line.strip()))
    return np.array(timestamps, dtype=np.float64)


# ==================== VISUAL ODOMETRY ====================
def calculate_visual_odometry(p0, p1):
    if len(p0) < MIN_FEATURES:
        return None, None, None

    E, mask = cv.findEssentialMat(
        p0, p1, K,
        method=cv.RANSAC,
        prob=0.999,
        threshold=ESS_RANSAC_THRESH
    )

    if E is None:
        return None, None, None

    inliers, R, t, mask_pose = cv.recoverPose(E, p0, p1, K, mask=mask)

    if inliers < MIN_INLIERS:
        return None, None, None

    t = t / (np.linalg.norm(t) + 1e-12)

    return R, t, mask_pose


# ==================== TRACKING + VISUALIZATION ====================
def process_pair(imgA, imgB, frame_idx):
    global LAST_POSE_R, LAST_POSE_t

    grayA = cv.cvtColor(imgA, cv.COLOR_BGR2GRAY)
    grayB = cv.cvtColor(imgB, cv.COLOR_BGR2GRAY)

    p0 = cv.goodFeaturesToTrack(
        grayA, maxCorners=1500, qualityLevel=0.01, minDistance=7
    )

    if p0 is None:
        return False

    p0 = p0.reshape(-1, 2).astype(np.float32)

    p1, st, err = cv.calcOpticalFlowPyrLK(grayA, grayB, p0, None)
    if p1 is None:
        return False

    st = st.reshape(-1)
    err = err.reshape(-1)

    valid = (st == 1)
    if valid.sum() == 0:
        return False

    err_th = np.mean(err[valid]) + 2 * np.std(err[valid])
    valid = valid & (err <= max(err_th, 12.0))

    p0 = p0[valid]
    p1 = p1.reshape(-1, 2)[valid]

    R, t, mask_vo = calculate_visual_odometry(p0, p1)
    if R is None:
        return False

    LAST_POSE_R = R.copy()
    LAST_POSE_t = t.copy()

    # ==================== LINE MATCHING VISUALIZATION ====================
    h1, w1 = imgA.shape[:2]
    h2, w2 = imgB.shape[:2]
    H = max(h1, h2)
    W = w1 + w2

    combined = np.zeros((H, W, 3), dtype=np.uint8)
    combined[:h1, :w1] = imgA
    combined[:h2, w1:w1 + w2] = imgB

    mask_vo = mask_vo.reshape(-1)

    max_draw = min(200, len(p0))
    for i in range(max_draw):
        if mask_vo[i] != 1:
            continue

        x0, y0 = p0[i]
        x1, y1 = p1[i]

        pt0 = (int(x0), int(y0))
        pt1 = (int(x1 + w1), int(y1))

        # Red match line
        cv.line(combined, pt0, pt1, (0, 0, 255), 1)
        cv.circle(combined, pt0, 3, (0, 255, 0), -1)
        cv.circle(combined, pt1, 3, (255, 0, 0), -1)

    save_path = os.path.join(OUTPUT_FOLDER, f"tracking_{frame_idx:05d}.png")
    cv.imwrite(save_path, combined)

    return True


# ==================== MAIN ====================
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    images = sorted(glob.glob(os.path.join(DATASET_PATH, "*.png")))
    timestamps = load_kitti_timestamps(TIMESTAMP_FILE)

    assert len(images) <= len(timestamps), "Timestamp mismatch!"

    with open(POSE_FILE, "w") as f:
        f.write("# t_i t_j\n# R (3x3)\n# t (unit direction)\n\n")

    for i in range(min(MAX_FRAMES, len(images) - 1)):
        imgA = cv.imread(images[i])
        imgB = cv.imread(images[i + 1])

        if imgA is None or imgB is None:
            continue

        ok = process_pair(imgA, imgB, i)
        if not ok:
            print(f"Frame {i}->{i+1}: VO failed")
            continue

        with open(POSE_FILE, "a") as f:
            f.write(f"{timestamps[i]:.9f} {timestamps[i+1]:.9f}\n")
            for row in LAST_POSE_R:
                f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
            f.write(" ".join(f"{v:.6f}" for v in LAST_POSE_t.flatten()) + "\n\n")

        print(f"Frame {i}->{i+1} | Δt = {timestamps[i+1] - timestamps[i]:.6f}s")

    print("\n✅ Finished VO with line matching & timestamps")
    print("Saved poses to:", POSE_FILE)


# ==================== RUN ====================
if __name__ == "__main__":
    main()
