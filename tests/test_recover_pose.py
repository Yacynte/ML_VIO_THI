import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R_sci

def test_convention():
    # 1. Define a camera matrix
    K = np.array([[500, 0, 320],
                  [0, 500, 240],
                  [0, 0, 1]], dtype=np.float64)

    # 2. Define a relative motion: Move forward and turn right
    # Camera 0 is at origin
    # Camera 1 moves forward (Z=1) and turns right (Yaw = -10 degrees)
    # Turn right is CW about Y-up, but in camera frame (Y-down) it is CCW about Y.
    # Let's say Yaw = +10 deg in Camera Frame (X moves towards -Z)
    yaw_deg = 10
    yaw_rad = np.deg2rad(yaw_deg)
    
    # R_prev_curr: rotation from Curr back to Prev
    # If Curr is rotated by +yaw relative to Prev: P_prev = R_yaw * P_curr
    r_mat = R_sci.from_euler('y', yaw_rad).as_matrix()
    t_in_prev = np.array([[0], [0], [1]], dtype=np.float64) # Curr origin in Prev
    
    # 3. Create synthetic points in 3D
    pts_3d = np.array([
        [1, 1, 5],
        [-1, 1, 5],
        [1, -1, 5],
        [-1, -1, 5],
        [2, 0, 10],
        [-2, 0, 10],
        [0, 2, 8],
        [0, -2, 8]
    ], dtype=np.float64)
    
    # Project to Camera 0 (Prev)
    pts_prev_px, _ = cv.projectPoints(pts_3d, np.zeros(3), np.zeros(3), K, None)
    pts_prev_px = pts_prev_px.reshape(-1, 2)
    
    # Project to Camera 1 (Curr)
    # P_curr = R^T (P_prev - t)
    pts_3d_curr = (r_mat.T @ (pts_3d.T - t_in_prev)).T
    pts_curr_px, _ = cv.projectPoints(pts_3d_curr, np.zeros(3), np.zeros(3), K, None)
    pts_curr_px = pts_curr_px.reshape(-1, 2)
    
    # 4. Use Essential Matrix and recoverPose
    E, mask = cv.findEssentialMat(pts_prev_px, pts_curr_px, K, method=cv.RANSAC)
    
    print("--- Testing recoverPose(E, pts_curr, pts_prev) ---")
    retval1, R_est1, t_est1, _ = cv.recoverPose(E, pts_curr_px, pts_prev_px, K)
    print("Estimated R:\n", np.round(R_est1, 4))
    print("Estimated t:", t_est1.flatten())
    
    print("\n--- Testing recoverPose(E, pts_prev, pts_curr) ---")
    retval2, R_est2, t_est2, _ = cv.recoverPose(E, pts_prev_px, pts_curr_px, K)
    print("Estimated R:\n", np.round(R_est2, 4))
    print("Estimated t:", t_est2.flatten())

    print("\nTrue R (Prev <- Curr):\n", np.round(r_mat, 4))
    print("True t (Curr in Prev):", t_in_prev.flatten())

    # Check if R_est is R_prev_curr or R_curr_prev
    diff_normal = np.linalg.norm(R_est - r_mat)
    diff_trans = np.linalg.norm(R_est - r_mat.T)
    
    if diff_normal < diff_trans:
        print("\nCONCLUSION: R is R_prev_curr (P_prev = R*P_curr + t)")
    else:
        print("\nCONCLUSION: R is R_curr_prev (P_curr = R*P_prev + t)")

if __name__ == "__main__":
    test_convention()
