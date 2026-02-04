import cv2
import numpy as np
import matplotlib.pyplot as plt

def shi_tomasi_klt_tracking(img_a_gray, img_b_gray, frame_index):
    # Shi-Tomasi features
    feature_params = dict(maxCorners=500, qualityLevel=0.01, minDistance=7, blockSize=7, useHarrisDetector=True, k=0.04)
    p0 = cv2.goodFeaturesToTrack(img_a_gray, mask=None, **feature_params)

    if p0 is None:
        print(f"No features in frame {frame_index}.")
        return False

    p0 = p0.reshape(-1, 2).astype(np.float32)

    # Optical flow tracking
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    p1, st, err = cv2.calcOpticalFlowPyrLK(img_a_gray, img_b_gray, p0, None, **lk_params)

    if p1 is None:
        print(f"KLT failed for frame {frame_index}.")
        return False

    st = st.reshape(-1)
    err = err.reshape(-1) if err is not None else np.zeros_like(st)

    success = True
    # Basic valid mask
    valid = (st == 1)
    if valid.sum() == 0:
        print(f"No valid tracked points at frame {frame_index}.")
        success = False

    # Remove bad error points (2*std)
    e_th = np.mean(err[valid]) + 2 * np.std(err[valid])
    valid = valid & (err <= max(e_th, 12.0))

    p0_tracked = p0[valid]
    p1_tracked = p1.reshape(-1, 2)[valid]

    
    return p0_tracked, p1_tracked, success


def sift_flann_matcher(img1, img2, grid_rows=2, grid_cols=4):
    # 1. Load images in grayscale
    if img1 is None or img2 is None:
        print("Error loading images.")
        return

    # 2. Initialize SIFT detector
    # Note: You can swap this with cv2.ORB_create() for speed
    sift = cv2.SIFT_create()

    # 3. Detect and Compute Keypoints and Descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 4. FLANN parameters (Fast Library for Approximate Nearest Neighbors)
    # KD-Tree index is efficient for SIFT/SURF
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50) # Higher checks = more precision, slower

    # If using ORB, use LSH index instead:
    # index_params= dict(algorithm = 6, table_number = 6, key_size = 12, multi_probe_level = 1)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 5. Matching
    # k=2 means we want the top 2 matches for every descriptor to apply the Ratio Test
    matches = flann.knnMatch(des1, des2, k=2)

    # 6. Apply Lowe's Ratio Test
    # We only keep matches where the best match is significantly better (0.7 distance)
    # than the second best match.
    good_matches = []
    outlier_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
        else:
            outlier_matches.append(m)

    # 7. Geometric Verification (RANSAC)
    # We need at least 4 matches to calculate a Homography
    MIN_MATCH_COUNT = 10

    spread_matches, outlier_matches = get_spatially_distributed_matches( good_matches, outlier_matches, kp1, img1.shape, grid_rows=grid_rows, grid_cols=grid_cols )

    # print(f"Matches before Ratio Test: {len(matches)}")
    # print(f"Matches after Ratio Test: {len(good_matches)}")
    # print(f"Spatially distributed matches: {len(spread_matches)}")
    if len(spread_matches) > MIN_MATCH_COUNT:
        # Extract locations of good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in spread_matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in spread_matches]).reshape(-1, 2)
    else:
        print(f"Not enough matches are found - {len(spread_matches)}/{MIN_MATCH_COUNT}")
        return None, None, None, None, None, None, False
    
    return src_pts, dst_pts, kp1, kp2, spread_matches, outlier_matches, True




def get_spatially_distributed_matches(good_matches, outlier_matches, keypoints, img_shape, grid_rows=2, grid_cols=4):
    """
    Filters matches to ensure they are spread out across the image using a grid.
    
    Args:
        matches: List of cv2.DMatch objects (after Ratio Test).
        keypoints: List of cv2.KeyPoint objects from the query image.
        img_shape: Tuple (height, width) of the query image.
        grid_rows: Number of rows to divide the image into.
        grid_cols: Number of columns to divide the image into.
        
    Returns:
        List of cv2.DMatch objects distributed across the grid.
    """
    h, w = img_shape[:2]
    
    # Calculate the size of each grid cell
    cell_h = h / grid_rows
    cell_w = w / grid_cols
    
    # Dictionary to store matches per cell: key=(row, col), value=[match]
    match_grid = {}
    
    for m in good_matches:
        # Get the location of the keypoint in the query image
        # m.queryIdx refers to the index in the 'keypoints' list passed to this function
        pt = keypoints[m.queryIdx].pt
        
        # Determine which grid cell this point belongs to
        row = int(pt[1] / cell_h)
        col = int(pt[0] / cell_w)
        
        # Clamp to edges to handle points exactly on the boundary
        row = min(row, grid_rows - 1)
        col = min(col, grid_cols - 1)
        
        if (row, col) not in match_grid:
            match_grid[(row, col)] = []
        
        match_grid[(row, col)].append(m)
    
    distributed_matches = []
    # Iterate over each cell and pick the best match
    for key in match_grid:
        cell_matches = match_grid[key]
        
        # Sort by distance (lower is better) to find the highest quality match in this cell
        cell_matches.sort(key=lambda x: x.distance)
        
        # Select the top 1 match per cell. 
        # You can increase this slice (e.g., [:2]) if you want higher density.
        num_matches = min(30, len(cell_matches))
        distributed_matches.extend(cell_matches[:num_matches])
        outlier_matches.extend(cell_matches[num_matches:num_matches*2])

    # for m in distributed_outlier_matches:
    #     outlier_matches.append(m)
    return distributed_matches, outlier_matches


def recover_pose_robust(img1, img2, kp1, kp2, matches, K, dist_coeffs=None):
    min_inliers = 30
    min_inlier_ratio = 0.3
    """
    Robustly recovers pose by handling distortion and tuning RANSAC.
    
    Args:
        img1, img2: Input images (for visualization).
        kp1, kp2: Keypoints from SIFT/ORB.
        matches: List of DMatch objects (your 1000+ spread matches).
        K: Camera Matrix (3x3).
        dist_coeffs: Distortion coefficients (1x5). If None, assumes 0.
    """
    
    # 1. Extract point coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 2. Undistort Points (Crucial for wide angle/GoPro)
    # Instead of undistorting the whole image (slow), we just undistort the points.
    if dist_coeffs is not None:
        pts1 = cv2.undistortPoints(pts1, K, dist_coeffs, P=K)
        pts2 = cv2.undistortPoints(pts2, K, dist_coeffs, P=K)

    # 3. Find Essential Matrix with Relaxed Threshold
    # threshold=3.0 allows for some localization error in high-res images
    E, mask = cv2.findEssentialMat(
        pts1, 
        pts2, 
        K, 
        method=cv2.RANSAC, 
        prob=0.999, 
        threshold=3.0 
    )

    if E is None:
        print("Essential Matrix calculation failed.")
        return

    # 4. Recover Pose
    # This checks for Cheirality (points must be in front of the camera)
    inlier_count, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

    total_matches = len(matches)
    inlier_ratio = inlier_count / total_matches

    # We need BOTH enough points AND a high enough confidence ratio
    if inlier_count >= min_inliers and inlier_ratio >= min_inlier_ratio:
        success = True
    else:
        success = False

    # 5. Filter Inliers
    # mask_pose contains the final inliers after RANSAC + Cheirality check
    matches_inliers = []
    matches_outliers = []
    
    for i, m in enumerate(matches):
        if mask_pose[i]:
            matches_inliers.append(m)
        else:
            matches_outliers.append(m)

    print(f"Total Matches: {len(matches)}")
    print(f"Inliers recovered: {len(matches_inliers)} ({len(matches_inliers)/len(matches):.1%})")

    # 6. Visualization
    # Draw Inliers in GREEN, Outliers in RED
    out_img = cv2.drawMatches(
        img1, kp1, 
        img2, kp2, 
        matches_inliers, 
        None, 
        matchColor=(0, 255, 0),    # Green for inliers
        singlePointColor=None,
        flags=2
    )
    
    # Overlay outliers in Red (optional, to see what failed)
    # We draw outliers on top to see if they are clustered in specific areas (e.g., edges)
    # out_img = cv2.drawMatches(
    #     img1, kp1, 
    #     img2, kp2, 
    #     matches_outliers, 
    #     out_img, 
    #     matchColor=(0, 0, 255),    # Red for outliers
    #     singlePointColor=None,
    #     flags=1 # Draw over existing
    # )

    # plt.figure(figsize=(15, 8))
    # plt.imshow(out_img)
    # plt.title("Green = Inliers (Used for Pose), Red = Outliers (Rejected)")
    # plt.axis('off')
    # plt.show()

    # cv2.imshow("Pose Recovery Inliers vs Outliers", out_img)
    # cv2.waitKey(1)
    return R, t, success



def decomp_essential_mat(E, pts1, pts2, K, dist_coeffs=None):
    """
    Decompose the Essential matrix

    Parameters
    ----------
    E (ndarray): Essential matrix
    q1 (ndarray): The good keypoints matches position in i-1'th image
    q2 (ndarray): The good keypoints matches position in i'th image

    Returns
    -------
    right_pair (list): Contains the rotation matrix and translation vector
    """
    # Instead of undistorting the whole image (slow), we just undistort the points.
    if dist_coeffs is not None:
        pts1 = cv2.undistortPoints(pts1, K, dist_coeffs, P=K).reshape(-1, 2)
        pts2 = cv2.undistortPoints(pts2, K, dist_coeffs, P=K).reshape(-1, 2)

    def sum_z_cal_relative_scale(R, t):
        # Get the transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        P0 = T[:3, :]
        # Make the projection matrix
        P = np.matmul(np.concatenate((K, np.zeros((3, 1))), axis=1), T)
        # Triangulate the 3D points
        hom_Q1 = cv2.triangulatePoints(np.float32(P0), np.float32(P), np.float32(pts1.T), np.float32(pts2.T))
        # Also seen from cam 2
        hom_Q2 = np.matmul(T, hom_Q1)
        # Un-homogenize
        uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
        uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

        # Find the number of points there has positive z coordinate in both cameras
        sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
        sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

        return sum_of_pos_z_Q1 + sum_of_pos_z_Q2

        # Decompose the essential matrix
    R1, R2, t = cv2.decomposeEssentialMat(E)
    t = np.squeeze(t)
    
    # Make a list of the different possible pairs
    pairs = [[R1, -t], [R1, t], [R2, t], [R2, -t]]

    # Check which solution there is the right one
    z_sums = []
    relative_scales = []
    for R, t in pairs:
        z_sum = sum_z_cal_relative_scale(R, t)
        z_sums.append(z_sum)

    # Select the pair there has the most points with positive z coordinate
    right_pair_idx = np.argmax(z_sums)
    num_matches = z_sums[right_pair_idx] // 2
    #print(right_pair_idx)
    right_pair = pairs[right_pair_idx]
    R1, t = right_pair

    return R1, t.reshape(3, 1), num_matches, True

def get_pose(q1, q2, K, dist_coeffs=None):
    # z_c is the distance between the center of the camera and the center of the optitrack markers. This is an approximate that was measured using a ruler
    """
    Calculates the transformation matrix

    Parameters
    ----------
    q1 (ndarray): The good keypoints matches position in i-1'th image
    q2 (ndarray): The good keypoints matches position in i'th image

    Returns
    -------
    transformation_matrix (ndarray): The transformation matrix
    """
    # Essential matrix
    E, _ = cv2.findEssentialMat(q1, q2, K, method=0, threshold=0.1)

    # Decompose the Essential matrix into R and t
    R, t = decomp_essential_mat(E, q1, q2, K, dist_coeffs)

    # Get transformation matrix
    # transformation_matrix = self._form_transf(R, np.squeeze(t))
    return R, t, True
