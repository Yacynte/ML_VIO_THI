# VO and IMU Processing 

## Goal
Create a robust, standalone script [VO_with_IMU.py] and intergration script [imu_processor_v2.py] to process IMU and VO data from KITTI and Urbaning datasets. The goal is to produce accurate dead-reckoning trajectories by correctly integrating accelerometer and gyroscope data, visual odometry and sensor fusion using a complementary filter.

## Features
### IMU Processing
1.  **Modular Data Loading**:
    *   **KITTI**: Support parsing OXTS text files and timestamps.
    *   **Urbaning**: Support parsing JSON files.
    *   **KITTI MAT**: (Optional) Support loading from .mat files if needed (from previous code).

2.  **State Estimation (Dead Reckoning)**:
    *   Uses a `State` class to track Position (p), Velocity (v), and Orientation (q).
    *   Implements Strapdown Inertial Navigation equations.
    *   **Orientation**: Updates using quaternion integration of angular rates.
    *   **Velocity**: Updates using gravity-compensated acceleration in the World frame.
    *   `v_dot = R_wb * a_body + g_world`
    *   **Position**: Standard kinematic integration.

3. **Mathematical Formulation**
*   **Coordinate Frame**: World Frame (ENU or defined by initial pose).
    *   Gravity: `[0, 0, -9.81]` m/s².
*   **Input**:
    *   Acceleration `a_b` (m/s²) from IMU (Specific Force).
    *   Angular Velocity `w_b` (rad/s) from IMU.
*   **Process**:
    1.  Initialize state (`p0`, `v0`, `q0`).
    2.  For each timestep `dt`:
        *   Update Orientation: `q_new = q_old * exp(w_b * dt)`.
        *   Rotate Acceleration: `a_world = R(q_new) * a_b`.
        *   Remove Gravity: `a_net = a_world + g_vector`.
        *   Update Velocity: `v_new = v_old + a_net * dt`.
        *   Update Position: `p_new = p_old + v_old * dt + 0.5 * a_net * dt^2`.

4.  **Visualization**:
    *   3D Trajectory Plot (matplotlib).
    *   3D Trajectory Plot (matplotlib).
    *   2D Top-down view (selectable via flag).
    *   Comparison with GPS/Ground Truth (if available).
    *   **Integration API**: Function [get_processed_imu_data] to return data for VIO.
    *   **Ground Truth Export**: API to return ground truth trajectory for verification in other scripts.

### VO Processing
1. **Modular Data Loading**:
    *   **KITTI**: Support parsing OXTS text files and timestamps.
    *   **Urbaning**: Support parsing JSON files.
    *   **KITTI MAT**: (Optional) Support loading from .mat files if needed (from previous code).

2. **VO Processing**:
    *   **Feature Detection**: Use ORB or SIFT for feature detection.
    *   **Feature Matching**: Use FLANN or BFMatcher for feature matching.
    *   **Essential Matrix Estimation**: Use `cv.findEssentialMat` with RANSAC for robust estimation.
    *   **Pose Recovery**: Use `cv.recoverPose` to recover the relative pose between frames. OR use `cv.decomposeEssentialMat` to decompose the essential matrix into 4 possible     rotation and translation. Then select the correct one based on the number of points in front of the camera and test which one gives the best result with `cv.triangulatePoints`.

3. **VIO Robustness and Precision**:
    *   **Optimize `findEssentialMat`**: 
        *   Uses `cv.USAC_MAGSAC` (since OpenCV >= 4.5 is available) for better outlier rejection.
        *   Reduce threshold from `6.0` to `1.0` or `0.5` for normalized coordinates.
    *   **Coordinate Transformation Fixes**:
        *   [VO_with_IMU.py] initialization to correctly compute the Camera-to-World orientation (ground truth in camera frame).
        *   Unify the translation update logic to ensure $P_{new} = P_{old} + R_{old} \cdot t_{new\_in\_old}$.


### VIO Integration
1. **VIO Integration in [VO_with_IMU.py]**:
    *   Update `imu_processor_v2` to get IMU and Ground Truth data.
    *   Use IMU to scale, drift correct and smooth the VO trajectory.
    *   Use complementary filter to fuse IMU and VO data.


### File Structure
*   [imu_processor_v2.py]: The single self-contained IMU and ground truth script.
*   [VO_with_IMU.py]: The VO and V0 - IMU intergration script.

## Installation Guide

Follow these steps to set up and run the project in your local environment.

### 1. Clone the repository

```bash
git clone https://github.com/Yacynte/ML_VIO_THI.git
cd ML_VIO_THI
```
### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the script

The user will be able to run:
```bash
python3 src/VO_with_IMU.py
```

