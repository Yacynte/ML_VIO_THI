import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R

# ----------------------------
# Utility functions
# ----------------------------

class TransformMatrix():
    def __init__(self, dataset_type='kitti', camera = 'front_left_camera', vehicle = 'vehicle2', cam_id="02"):
        self.dataset_type = dataset_type # 'kitti', 'urbaning', 'kitti_mat'
        self.camera = camera
        self.vehicle = vehicle
        self.cam_id = cam_id
        self.intrinsics = None
        self.distortion = None

    @staticmethod
    def make_T(R, t):
        """Create a 4x4 homogeneous transform from R (3x3) and t (3,)."""
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    
    @staticmethod
    def assemble_distortion_matrix(radial, tangential):
        """
        Assembles separate radial (1x3) and tangential (1x2) arrays
        into the specific 1x5 format OpenCV requires.
        
        Args:
            radial: [k1, k2, k3]
            tangential: [p1, p2]
        """
        # Ensure inputs are flat lists or arrays
        r = np.array(radial).flatten()
        t = np.array(tangential).flatten()
        
        # Sanity checks
        if len(r) != 3 or len(t) != 2:
            print(f"Error: Expected 3 radial and 2 tangential. Got {len(r)} and {len(t)}.")
            return None

        # CRITICAL: The order must be k1, k2, p1, p2, k3
        dist_coeffs = np.array([
            r[0],  # k1
            r[1],  # k2
            t[0],  # p1 (Tangential 1)
            t[1],  # p2 (Tangential 2)
            r[2]   # k3 (Radial 3)
        ], dtype=np.float32)
        
        return dist_coeffs.reshape(1, 5)

    def __load_calib_file(self, path):
        """Load a KITTI calibration file into a dictionary."""
        data = {}
        if self.dataset_type == 'urbaning':
            import json
            cam = self.vehicle + '_' + self.camera
            with open(path, 'r') as f:
                calib_data = json.load(f)[cam]
                self.intrinsics = np.array(calib_data['intrinsics']['IntrinsicMatrix']).squeeze().T
                distortion_radial = np.array(calib_data['intrinsics']['RadialDistortion']).squeeze()
                distortion_tangential = np.array(calib_data['intrinsics']['TangentialDistortion']).squeeze()
                self.distortion = self.assemble_distortion_matrix(distortion_radial, distortion_tangential)
                T_cam2car = np.array(calib_data['extrinsics']['cTv']).squeeze()
                # T_car2cam = np.linalg.inv(T_cam2car)
            return T_cam2car

        if self.dataset_type == 'kitti_mat':
            with open(path, "r") as f:
                for line in f:
                    if ":" not in line:
                        continue
                    key, value = line.strip().split(":", 1)
                    if key[-1] == self.cam_id[-1]:
                        data = np.fromstring(value, sep=" ").reshape(3,4)
            R = data[:3, :3]
            t = data[:3, 3]
            data = self.make_T(R, t)
            return data
        
        with open(path, "r") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, value = line.strip().split(":", 1)
                data[key] = np.fromstring(value, sep=" ")
        return data


    # ----------------------------
    # Load individual transforms
    # ----------------------------

    def __load_T_imu_to_velo(self, calib_imu_to_velo):
        data = self.__load_calib_file(calib_imu_to_velo)
        R = data["R"].reshape(3, 3)
        t = data["T"].reshape(3)
        return self.make_T(R, t)

    def __load_T_velo_to_cam0(self, calib_velo_to_cam):
        data = self.__load_calib_file(calib_velo_to_cam)
        R = data["R"].reshape(3, 3)
        t = data["T"].reshape(3)
        return self.make_T(R, t)


    def __load_cam_trans(self, path, cam_id=2):
        """
        cam_id: int, e.g., 0, 1, 2, 3
        """
        with open(path, "r") as f:
            for line in f:
                if f"R_{cam_id:02d}" in line:
                    key, value = line.strip().split(":", 1)
                    R = np.fromstring(value, sep=" ").reshape(3, 3)
                if f"T_{cam_id:02d}" in line:
                    key, value = line.strip().split(":", 1)
                    t = np.fromstring(value, sep=" ").reshape(3)
                if f"K_{cam_id:02d}" in line:
                    key, value = line.strip().split(":", 1)
                    self.intrinsics = np.fromstring(value, sep=" ").reshape(3, 3)
                if f"D_{cam_id:02d}" in line:
                    key, value = line.strip().split(":", 1)
                    self.distortion = np.fromstring(value, sep=" ").reshape(1, 5)
                # if f"R_rect_{cam_id:2d}" in line:
                #     key, value = line.strip().split(":", 1)
                #     R_rect = np.fromstring(value, sep=" ").reshape(3, 3)
                # if f"P_rect_{cam_id:2d}" in line:
                #     key, value = line.strip().split(":", 1)
                #     P_rect = np.fromstring(value, sep=" ").reshape(3, 4)
        T = self.make_T(R, t)
        return T


    


    # ----------------------------
    # Main chaining logic
    # ----------------------------

    def compute_T_imu_to_cam(self, calib_imu_to_velo, calib_velo_to_cam, calib_cam_to_cam=None, cam_id=0):

        if self.dataset_type == 'kitti_mat':
            data = self.__load_calib_file(calib_imu_to_velo)
            return data
        # IMU -> Velodyne
        T_imu_velo = self.__load_T_imu_to_velo(calib_imu_to_velo)

        # Velodyne -> Cam0
        T_velo_cam0 = self.__load_T_velo_to_cam0(calib_velo_to_cam)
        # IMU -> Cam0
        T_imu_cam0 = T_velo_cam0 @ T_imu_velo
        # Cam -> CamX
        T_cam0_camX = self.__load_cam_trans(calib_cam_to_cam, cam_id)

        # IMU -> CamX
        T_imu_camX = T_cam0_camX @ T_imu_cam0
        return T_imu_camX

    def compute_T_cam_to_imu_urbaning(self, calib_cam_to_imu):
        T_imu_cam = self.__load_calib_file(calib_cam_to_imu)
        return T_imu_cam


    # def get_camera_intrinsics(self, calib_cam_to_cam, cam_id="00"):
    #     data = self.__load_calib_file(calib_cam_to_cam)
    #     P = data[f"P_{cam_id}"].reshape(3, 4)
    #     K = P[:3, :3]
    #     return K

    # def get_T_imu_to_cam(self, imu2velo_path, velo2cam_path, cam2cam_path=None, cam_id="00"):
    #     T_imu_cam = self.compute_T_imu_to_cam(imu2velo_path, velo2cam_path, cam2cam_path, cam_id)
    #     return T_imu_cam

def imu_pose_to_camera_pose(R_w_imu, p_w_imu, T_imu_cam):
    """
    Convert IMU pose in world frame to camera pose in world frame.
    """
    if R_w_imu.shape != (3, 3) and len(R_w_imu) == 3:
        r_w_imu_ = R.from_euler('xyz', R_w_imu, degrees=False)    # Roll-Pitch-Yaw
        R_w_imu = r_w_imu_.as_matrix()
    T_w_imu = np.eye(4)
    T_w_imu[:3, :3] = R_w_imu
    T_w_imu[:3, 3] = p_w_imu

    # Correct order: T_world_cam = T_world_imu * T_imu_cam
    # Assuming T_imu_cam is T_camera_to_imu (cTv in urbaning)
    # We want T_imu_to_camera = inv(T_imu_cam)
    T_imu_to_cam = np.linalg.inv(T_imu_cam)
    T_w_cam = T_w_imu @ T_imu_to_cam
    
    rot_mat = T_w_cam[:3, :3]
    rot_vec = R.from_matrix(rot_mat).as_euler('xyz', degrees=False)
    position = T_w_cam[:3, 3]
    return rot_vec, position


def camera_pose_to_imu_pose(R_w_cam, p_w_cam, T_imu_cam):
    """
    Convert camera pose in world frame to IMU pose in world frame.
    """
    T_cam_imu = np.linalg.inv(T_imu_cam)

    if R_w_cam.shape != (3, 3) and len(R_w_cam) == 3:
        r_w_cam_ = R.from_euler('xyz', R_w_cam, degrees=False)    # Roll-Pitch-Yaw
        R_w_cam = r_w_cam_.as_matrix()
    T_w_cam = np.eye(4)
    T_w_cam[:3, :3] = R_w_cam
    T_w_cam[:3, 3] = p_w_cam

    # Correct order: T_world_imu = T_world_cam * T_cam_imu
    # Where T_cam_imu is T_imu_to_cam = inv(T_imu_cam)
    # T_world_imu = T_world_cam @ T_imu_cam
    T_w_imu = T_w_cam @ T_imu_cam
    
    rot_mat = T_w_imu[:3, :3]
    rot_vec = R.from_matrix(rot_mat).as_euler('xyz', degrees=False)
    position = T_w_imu[:3, 3]
    return rot_vec, position

# ----------------------------
# CLI entry point
# ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute KITTI IMU -> Camera transformation"
    )
    parser.add_argument("--imu2velo", required=True, help="calib_imu_to_velo.txt")
    parser.add_argument("--velo2cam", required=True, help="calib_velo_to_cam.txt")
    parser.add_argument("--cam2cam", default=None, help="calib_cam_to_cam.txt")
    parser.add_argument("--cam_id", default="02", help="Camera ID: 00,01,02,03")

    args = parser.parse_args()

    transformer = TransformMatrix(dataset_type='kitti')
    T_imu_cam = transformer.get_T_imu_to_cam(imu2velo_path=args.imu2velo,
                                             velo2cam_path=args.velo2cam,
                                             cam2cam_path=args.cam2cam,
                                             cam_id=args.cam_id)

    np.set_printoptions(precision=6, suppress=True)
    print(f"\nT_imu_cam{args.cam_id} =\n{T_imu_cam}\n")
