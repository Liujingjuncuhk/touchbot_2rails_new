
import pyvista as pv
import subprocess
import numpy as np
import os
import open3d as o3d
import sys
import time
_ROYALE_PYTHON_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "libroyale-5.16.1.3725-LINUX-x86-64Bit", "python"
)
if _ROYALE_PYTHON_PATH not in sys.path:
    sys.path.insert(0, _ROYALE_PYTHON_PATH)

try:
    from roypypack import roypy
except ImportError:
    import roypy

class CameraDriver:
    def __init__(self) -> None:
        # dx, dy, dz are calibrated
        
        self.initialize()
        self.T_world_cam = np.array([[0, -1, 0, 0.00662],
                                     [0, 0, 1, -0.13864],
                                     [-1, 0, 0, 0.1072],
                                     [0, 0, 0, 1]])
    def initialize(self):
        import sys
        sys.path.append('/home/jingjun/Desktop/libroyale-5.10.0.2751-LINUX-x86-64Bit/python')
        import roypy
        from sample_camera_info import print_camera_info
        from roypy_sample_utils import CameraOpener, add_camera_opener_options, select_use_case
        from roypy_platform_utils import PlatformHelper
        import argparse
        import queue
        from sample_3d import MyListener
        platformhelper = PlatformHelper()
        parser = argparse.ArgumentParser (usage = __doc__)
        add_camera_opener_options(parser)
        # parser.add_argument ("--frames", type=int, required=True, help="duration to capture data (number of frames)")
        # parser.add_argument ("--output", type=str, required=True, help="filename to record to")
        # parser.add_argument ("--skipFrames", type=int, default=0, help="frameSkip argument for the API method")
        # parser.add_argument ("--skipMilliseconds", type=int, default=0, help="msSkip argument for the API method")
        options = parser.parse_args()
        opener = CameraOpener (options)
        self.cam = opener.open_hardware_camera()
        use_case = 'Mode_5_15fps'
        self.cam.setUseCase(use_case)

        assert(self.cam.getId() == '8434-235A-2730-6630')
        self.q = queue.Queue()
        self.l = MyListener(self.q)
        self.cam.registerDataListener(self.l)
        self.cam.startCapture()
        time.sleep(1)
        print("initializing done")

    def take_data(self, q):
        if len(q.queue) == 0:
            # print("no data")
            data = q.get(True, 1)
        else:
            for i in range (0, len (q.queue)):
                data = q.get(True, 1)
        data = data[np.all(data != 0, axis=1)]
        return data

    def filter_data(self, pts, x_range=None, y_range=None, z_range=None):

        if x_range is not None:
            pts = pts[(pts[:, 0] >= x_range[0]) & (pts[:, 0] <= x_range[1])]
        if y_range is not None:
            pts = pts[(pts[:, 1] >= y_range[0]) & (pts[:, 1] <= y_range[1])]
        if z_range is not None:
            pts = pts[(pts[:, 2] >= z_range[0]) & (pts[:, 2] <= z_range[1])]

        return pts

    def stop_capture(self):
        self.cam.stopCapture()
        # self.cam2.stopCapture()

    def take_ply(self):
        # run cmd : sudo python /home/jingjun/Desktop/libroyale-5.10.0.2751-LINUX-x86-64Bit/python/sample_record_rrf.py --frames 11 --output /home/jingjun/Desktop/libroyale-5.10.0.2751-LINUX-x86-64Bit/testrecord.rrf --skipFrames 10
        subprocess.run(["sudo", "python", "/home/jingjun/Desktop/libroyale-5.10.0.2751-LINUX-x86-64Bit/python/sample_record_rrf.py", "--frames", "11", "--output", "/home/jingjun/Desktop/libroyale-5.10.0.2751-LINUX-x86-64Bit/testrecord.rrf", "--skipFrames", "10"])

        # run cmd: sudo python /home/jingjun/Desktop/libroyale-5.10.0.2751-LINUX-x86-64Bit/python/sample_export_ply.py --frame 0 --output testply --rrf /home/jingjun/Desktop/libroyale-5.10.0.2751-LINUX-x86-64Bit/testrecord.rrf
        subprocess.run(["sudo", "python", "/home/jingjun/Desktop/libroyale-5.10.0.2751-LINUX-x86-64Bit/python/sample_export_ply.py", "--frame", "0", "--output", "testply", "--rrf", "/home/jingjun/Desktop/libroyale-5.10.0.2751-LINUX-x86-64Bit/testrecord.rrf"])

        input_file = "/home/jingjun/Desktop/github_codes/touchBot_2rails/testply0.ply"
        return input_file
    
    def process_ply(self, input_file):
        pcd = o3d.io.read_point_cloud(input_file)
        point_cloud_in_numpy = np.asarray(pcd.points)
        point_cloud_in_numpy = point_cloud_in_numpy*1000
        point_cloud_in_numpy = self.transform_coordinate(point_cloud_in_numpy)
        return point_cloud_in_numpy

    def transform_coordinate(self, point_cloud_in_numpy):
        R = self.T_world_cam[:3, :3]
        t = self.T_world_cam[:3, 3]
        global_pts = R @ point_cloud_in_numpy.T + t.reshape(-1, 1)
        global_pts = global_pts.T

        return global_pts
    
    def get_touch_region_pts(self, point_cloud_in_numpy):
        # get the touch region
        # the touch region is defined as the region where the z value is less than 10
        pt_touch = point_cloud_in_numpy[np.where(point_cloud_in_numpy[:,0] > self.touchregion[0][0])]
        pt_touch = pt_touch[np.where(pt_touch[:,0] < self.touchregion[0][1])]
        pt_touch = pt_touch[np.where(pt_touch[:,1] > self.touchregion[1][0])]
        pt_touch = pt_touch[np.where(pt_touch[:,1] < self.touchregion[1][1])]
        pt_touch = pt_touch[np.where(pt_touch[:,2] > self.touchregion[2][0])]
        pt_touch = pt_touch[np.where(pt_touch[:,2] < self.touchregion[2][1])]
        # print max z - min z
        # print(max(pt_touch[:,2]) - min(pt_touch[:,2]))
        return pt_touch
    
    def plot_touch_region(self, point_cloud_in_numpy):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_in_numpy)
        o3d.visualization.draw_geometries([pcd])

    def visualize_pts_axis(self, pts):
        # visualize the pts with coordinate axis
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        o3d.visualization.draw_geometries([pcd, axis])

    def fit_circle(self, points):
    # Extract x and y coordinates
        x = points[:, 0]
        y = points[:, 1]
        # Setup the matrix A based on transformed equations: [2x 2y 1]
        A = np.column_stack((2*x, 2*y, np.ones(len(x))))
        
        # Setup the vector B which is [x^2 + y^2]
        B = x**2 + y**2
        # Solve the least squares problem
        sol, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
        a, b, c = sol

        # Calculate the center (a, b) and radius r
        center = [a, b]
        r = np.sqrt(c + a**2 + b**2)
        return center, r
    
    def get_circle_onSides(self, pts):
        # get the circle on the sides
        # get the points on the sides
        zmin = min(pts[:,2])
        zmax = max(pts[:,2])
        left_pts = pts[np.where(pts[:,2] <= zmin + 5)]
        right_pts = pts[np.where(pts[:,2] >= zmax - 5)]
        center_left, r_left = self.fit_circle(left_pts[:,0:2])
        center_right, r_right = self.fit_circle(right_pts[:,0:2])
        return center_left, r_left, center_right, r_right
    
if __name__ == "__main__":

    cam_driver = CameraDriver()
    data = cam_driver.take_data(cam_driver.q)
    data_global = cam_driver.transform_coordinate(data)
    data_global = cam_driver.filter_data(data_global, x_range=(-1, 1.0), y_range=(-1.0, 0.4), z_range=(0.0, 0.3))
    cam_driver.visualize_pts_axis(data_global)
    cam_driver.stop_capture()

