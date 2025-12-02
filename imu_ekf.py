import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as R
from imu import IMUDataLoader, get_imu_data, get_gps_coords
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

GRAVITY = np.array([0, 0, 9.80665])  # world z-up; adjust sign if needed

def skew(v):
    """Skew-symmetric matrix for cross product."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def quat_mul(q1, q2):
    """Hamilton product. Quaternions as [x,y,z,w] (scipy style)."""
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    return (r1 * r2).as_quat()

def quat_exp(phi):
    """Exponential map from rotation vector (3,) to quaternion (x,y,z,w)."""
    angle = np.linalg.norm(phi)
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = phi / angle
    return R.from_rotvec(axis * angle).as_quat()

def quat_log(q):
    """Log map from quaternion to rotation vector."""
    return R.from_quat(q).as_rotvec()

# WGS-84 Earth Parameters (Constants for ECEF to Local Frame Conversion)
A = 6378137.0         # Semi-major axis (equatorial radius) in meters
F = 1 / 298.257223563 # Flattening
E_SQUARED = 2 * F - F**2 # First eccentricity squared

def convert_to_local_enu(
    lon_deg: float, lat_deg: float, alt: float, lon_0_deg: float, lat_0_deg: float, alt_0: float) -> np.ndarray:
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


import numpy as np

# WGS-84 constants
A = 6378137.0                      # semi-major axis (m)
F = 1.0 / 298.257223563            # flattening
E_SQ = 2 * F - F**2                # first eccentricity squared

def lla_to_ecef(lon_deg, lat_deg, alt):
    """
    Convert LLA (lon, lat in degrees, alt in meters) to ECEF (meters).
    Accepts scalars or arrays. Returns shape (..., 3).
    """
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    alt = np.asarray(alt)

    # make everything arrays broadcastable
    lon, lat = np.broadcast_to(lon, np.shape(alt)), np.broadcast_to(lat, np.shape(alt))

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    N = A / np.sqrt(1 - E_SQ * sin_lat**2)   # prime vertical radius of curvature

    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1 - E_SQ) + alt) * sin_lat

    # stack into (...,3)
    return np.stack((x, y, z), axis=-1)


def ecef_to_ned_matrix(lon0_deg, lat0_deg):
    """
    Rotation matrix from ECEF to NED at the origin (lon0, lat0).
    R_ned_ecef @ (r - r0) -> vector in NED.
    """
    lon0 = np.deg2rad(lon0_deg)
    lat0 = np.deg2rad(lat0_deg)

    s_lat = np.sin(lat0)
    c_lat = np.cos(lat0)
    s_lon = np.sin(lon0)
    c_lon = np.cos(lon0)

    # ECEF -> NED rotation matrix
    R = np.array([
        [-s_lat * c_lon, -s_lat * s_lon,  c_lat],
        [-s_lon,          c_lon,         0.0 ],
        [-c_lat * c_lon, -c_lat * s_lon, -s_lat]
    ])
    return R


def convert_to_ned(lon_deg, lat_deg, alt, lon0_deg, lat0_deg, alt0):
    """
    Convert (lon, lat, alt) to NED coordinates (meters) relative to origin (lon0, lat0, alt0).
    Inputs can be scalars or arrays. Output shape: (3,) for single point or (N,3) for arrays.
    Order: lon_deg, lat_deg, alt (matches your earlier function).
    """
    # convert inputs to numpy arrays to handle vectorization
    lon = np.asarray(lon_deg)
    lat = np.asarray(lat_deg)
    alt = np.asarray(alt)

    single_input = False
    if lon.shape == () and lat.shape == () and alt.shape == ():
        # single scalar point
        single_input = True
        lon = lon.reshape((1,))
        lat = lat.reshape((1,))
        alt = alt.reshape((1,))

    # compute ECEF positions
    r = lla_to_ecef(lon, lat, alt)          # shape (N,3)
    r0 = lla_to_ecef(lon0_deg, lat0_deg, alt0).reshape((1,3))  # origin (1,3)

    # rotation matrix (ECEF -> NED) at origin
    R = ecef_to_ned_matrix(lon0_deg, lat0_deg)

    # delta vector(s)
    delta = r - r0  # broadcast (N,3) - (1,3) -> (N,3)

    # apply rotation: for each delta, ned = R @ delta
    # R shape (3,3), delta shape (N,3) -> result (N,3)
    ned = (R @ delta.T).T

    if single_input:
        return ned[0]   # (3,)
    return ned         # (N,3)


class ESEKF:
    def __init__(self, p0=None, v0=None, q0=None):
        # Nominal state
        self.p = np.zeros(3) if p0 is None else p0.copy()
        self.v = np.zeros(3) if v0 is None else v0.copy()
        self.v_prev = self.v.copy()  # for position update
        self.q = np.array([0,0,0,1.0]) if q0 is None else q0.copy()  # x,y,z,w
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)

        # Error-state covariance (15x15)
        self.P = np.eye(15) * 1e-3

        # Continuous white noises (tunable)
        sigma_acc = 0.1       # m/s^2 / sqrt(Hz)
        sigma_gyro = 0.01     # rad/s / sqrt(Hz)
        sigma_ba = 1e-3       # m/s^2 / sqrt(Hz)
        sigma_bg = 1e-4       # rad/s / sqrt(Hz)

        # Process noise covariance in continuous-time (for the noise inputs)
        self.Qc = np.zeros((12,12))
        self.Qc[0:3,0:3] = (sigma_acc**2) * np.eye(3)   # accel noise
        self.Qc[3:6,3:6] = (sigma_gyro**2) * np.eye(3)  # gyro noise
        self.Qc[6:9,6:9] = (sigma_ba**2) * np.eye(3)    # accel bias random walk
        self.Qc[9:12,9:12] = (sigma_bg**2) * np.eye(3)  # gyro bias random walk

    def predict(self, imu_omega, imu_acc, dt):
        """
        Propagate nominal state and covariance with one IMU sample.
        imu_omega: measured angular rate (rad/s) in body frame (3,)
        imu_acc: measured accel (m/s^2) in body frame (3,)
        dt: time step (s)
        """
        # ---- Nominal state propagation ----
        # Correct measurements for current bias estimates
        omega = imu_omega - self.bg
        acc = imu_acc - self.ba

        # Orientation: q <- q * Exp(omega * dt)
        dq = quat_exp(omega * dt)
        self.q = quat_mul(self.q, dq)
        self.q = self.q / np.linalg.norm(self.q)  # renormalize

        # Acceleration in world
        Rwb = R.from_quat(self.q).as_matrix()  # body->world
        acc_world = Rwb.dot(acc) - GRAVITY
        # acc_world = acc - GRAVITY
        # Velocity and position
        self.v = self.v + acc_world * dt
        self.p = self.p + self.v_prev * dt + 0.5 * acc_world * (dt**2)

        # Store the previous velocity (v_prev is needed for accurate position update)
        self.v_prev = self.v.copy()

        # ---- Covariance propagation (linearized error dynamics) ----
        # Build continuous-time F (15x15) and G (15x12) matrices (see Forster et al.)
        F = np.zeros((15,15))
        G = np.zeros((15,12))

        # error-state order: [dtheta(3), dp(3), dv(3), dba(3), dbg(3)]
        # d/dt dtheta = -dbg + noise_gyro  -> F[0:3, 12:15] = -I
        F[0:3, 12:15] = -np.eye(3)

        # dp/dt = dv
        F[3:6, 6:9] = np.eye(3)

        # dv/dt = -R * skew(acc) * dtheta - dba + noise_acc
        # Approximate: F[6:9, 0:3] = -R * skew(acc)
        F[6:9, 0:3] = -Rwb.dot(skew(acc))

        # dv/dt wrt dba
        F[6:9, 9:12] = -Rwb

        # dtheta/dt wrt dbg (already set)
        # G maps continuous noise vector [acc_noise(3), gyro_noise(3), ba_rw(3), bg_rw(3)]
        G[6:9, 0:3] = Rwb  # accel noise affects dv
        G[0:3, 3:6] = -np.eye(3)  # gyro noise affects dtheta
        G[9:12, 6:9] = np.eye(3)  # ba random walk
        G[12:15, 9:12] = np.eye(3) # bg random walk

        # --- Exact Discrete Process Noise Qd Calculation (Closed-Form Solution) ---
        
        # Define the 2n x 2n matrix M 
        # M = [ -F , G*Qc*G^T ]
        #     [  0 ,    F^T   ]
        n = 15
        
        G_Qc_GT = G.dot(self.Qc).dot(G.T)
        
        M = np.zeros((2 * n, 2 * n))
        M[0:n, 0:n] = -F
        M[0:n, n:2*n] = G_Qc_GT
        M[n:2*n, n:2*n] = F.T

        # Compute the Matrix Exponential of M * dt
        # The result is divided into 4 n x n blocks:
        # exp(M*dt) = [ A , B ]
        #             [ C , D ]
        A_B_C_D = expm(M * dt)
        
        # Extract the blocks
        A = A_B_C_D[0:n, 0:n]
        B = A_B_C_D[0:n, n:2*n]

        # Compute Phi (A block) and Qd (A^T * B)
        Phi = A.T # This is the exact discrete transition matrix: Phi = exp(F*dt)
        Qd = Phi.dot(B) # Exact discrete process noise

        # Covariance update: P <- Phi * P * Phi.T + Qd
        self.P = Phi.dot(self.P).dot(Phi.T) + Qd

    def inject_error_and_reset(self, delta_x):
        """
        Apply additive error-state correction delta_x (15,) to the nominal.
        Then reset error-state (we assume EKF stores error in state, applies correction and zeros it).
        delta_x: [dtheta(3), dp(3), dv(3), dba(3), dbg(3)]
        """
        dtheta = delta_x[0:3]
        dp = delta_x[3:6]
        dv = delta_x[6:9]
        dba = delta_x[9:12]
        dbg = delta_x[12:15]

        # Quaternion correction: q <- q * Exp(dtheta)
        dq = quat_exp(dtheta)
        self.q = quat_mul(self.q, dq)
        self.q = self.q / np.linalg.norm(self.q)

        self.p = self.p + dp
        self.v = self.v + dv
        self.ba = self.ba + dba
        self.bg = self.bg + dbg

    def update_position(self, z_p, R_meas):
        """
        Simple measurement update for absolute position measurement z_p (3,).
        R_meas: measurement covariance (3x3)
        Uses linearized measurement: z = p_nominal + δp + noise
        """
        # H maps error-state to measurement: z = p + δp => H = [0_{3x3}, I3, 0..., 0]
        H = np.zeros((3,15))
        H[:, 3:6] = np.eye(3)

        # Innovation
        z_pred = self.p
        y = z_p - z_pred

        S = H.dot(self.P).dot(H.T) + R_meas
        K = self.P.dot(H.T).dot(np.linalg.inv(S))

        delta_x = K.dot(y)  # 15x1

        # Inject error into nominal and reset small error (i.e., zero-mean)
        self.inject_error_and_reset(delta_x)

        # Reset covariance: P <- (I - K H) P (Joseph form recommended)
        I = np.eye(15)
        self.P = (I - K.dot(H)).dot(self.P).dot((I - K.dot(H))).T + K.dot(R_meas).dot(K.T)

# ---------------------------
# Minimal test with synthetic IMU
# ---------------------------
def plot_imu_trajectory(pose_ekf: np.ndarray, gps_pose: np.ndarray = None):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pose_ekf[:, 0], pose_ekf[:, 1], pose_ekf[:, 2], label='IMU Trajectory')
    if gps_pose is not None:
        ax.plot(gps_pose[:, 0], gps_pose[:, 1], gps_pose[:, 2], label='GPS Trajectory', linestyle='--')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('3D IMU Trajectory from INS')
    ax.legend()
    plt.show()

def synthetic_test():
    ekf = ESEKF()
    imu_folder = '2011_09_26_drive_0013_sync/2011_09_26/2011_09_26_drive_0013_sync/oxts'
    imu_data, gps_coords_ = get_imu_data(imu_folder)
    # gps_coords = get_gps_coords(gps_coords_)
    gps_coords = convert_to_ned(
        gps_coords_[:,0], gps_coords_[:,1], gps_coords_[:,2],
        gps_coords_[0,0], gps_coords_[0,1], gps_coords_[0,2])
    
    imu_pose = []
    dt = np.diff(imu_data[:, -1])
    N = imu_data.shape[0]

    # simulate zero motion but with gravity in body frame
    imu_omega = np.zeros(3)
    imu_acc = np.array([0.0, 0.0, 9.80665])  # stationary: accel senses gravity in body frame

    # feed imu chain, we expect position/velocity stay near zero if biases are zero and gravity removed in world frame
    for i in range(1, N):
        ekf.predict(imu_data[i, 3:6], imu_data[i, 0:3], dt[i-1])
        imu_pose.append(ekf.p)
    imu_pose = np.array(imu_pose)
    plot_imu_trajectory(imu_pose, gps_coords)    
    print("After predict (stationary): p=", ekf.p, "v=", ekf.v)
    # Now simulate a position measurement at origin with small noise -> should correct to zero
    z_p = np.zeros(3)
    R_meas = np.eye(3) * 1e-2
    ekf.update_position(z_p, R_meas)
    print("After position update: p=", ekf.p, "v=", ekf.v)
    print("Final covariance trace:", np.trace(ekf.P))

if __name__ == "__main__":
    synthetic_test()