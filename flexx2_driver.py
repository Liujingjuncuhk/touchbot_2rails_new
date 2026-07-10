import sys
import os
import queue
import time
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Royale path setup
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Listener  (one type, used for everything)
# ---------------------------------------------------------------------------

class _Listener(roypy.IDepthDataListener):
    """Puts each frame as a dict into a queue.

    Keys:
      'xyz'  — (H, W, 3) float32  x, y, z in metres
      'gray' — (H, W)    uint16   amplitude / IR equivalent
      'conf' — (H, W)    uint8    depth confidence
    """
    def __init__(self):
        super().__init__()
        self.queue = queue.Queue()

    def onNewData(self, data):
        pc   = data.npoints()                    # (H, W, 5)
        xyz  = pc[:, :, 0:3].astype(np.float32)
        gray = pc[:, :, 3].astype(np.uint16)
        conf = pc[:, :, 4].astype(np.uint8)
        self.queue.put({"xyz": xyz, "gray": gray, "conf": conf})

    def get_frame(self, timeout=3.0) -> dict:
        """Block until a frame arrives; return the most recent one."""
        frame = None
        deadline = time.time() + timeout
        while True:
            try:
                frame = self.queue.get(timeout=max(0.05, deadline - time.time()))
                if self.queue.empty():
                    break
            except queue.Empty:
                break
        if frame is None:
            raise TimeoutError("No frame received within timeout.")
        return frame


# ---------------------------------------------------------------------------
# Driver class
# ---------------------------------------------------------------------------

class Flexx2Driver:
    """Flexx2 ToF camera driver wrapping libroyale 5.16.x.

    Usage (simplest)::

        cam = Flexx2Driver()
        cam.connect()

        detections, img = cam.detect_tags()   # detect and show annotated IR

        xyz, gray, conf = cam.get_frame()     # raw frame data
        cam.stop()

    Parameters
    ----------
    camera_id : str or None
        Royale camera ID.  ``None`` opens the first connected camera.
    use_case : str
        Royale operation mode.  Default ``'MODE_9_5fps'``.
    access_code : str
        Activation code for Level 2/3 access.  Leave empty for Level 1.
    """

    def __init__(
        self,
        camera_id: str | None = None,
        use_case: str = "Mode_9_5fps",
        access_code: str = "",
    ):
        self._camera_id  = camera_id
        self._use_case   = use_case
        self._access_code = access_code
        self.T_cam_world = np.array([[ 0. ,      0. ,      1. ,     -0.119  ],
                                    [ 1. ,      0.  ,     0.  ,    -0.02518],
                                    [-0. ,     -1. ,     -0.  ,     0.01264],
                                    [ 0. ,      0.  ,     0.   ,    1.     ]])

        self.T_world_cam = np.array([[0, -1, 0, 0.04638],
                                     [0, 0, 1, -0.13564],
                                     [-1, 0, 0, 0.1072],
                                     [0, 0, 0, 1]])
        # All royale objects live here for the full lifetime of the driver.
        # Never set these to None while the camera may still be capturing —
        # that races with the C++ capture thread and causes heap corruption.
        self._manager  = None
        self._cam      = None
        self._listener = None   # created in connect()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self):
        """Open the camera and start streaming."""
        self._listener = _Listener()

        self._manager = roypy.CameraManager(self._access_code)
        cam_list = self._manager.getConnectedCameraList()

        if cam_list.size() == 0:
            raise RuntimeError("No Flexx2 cameras found.")

        cam_id = self._camera_id if self._camera_id else cam_list[0]
        self._cam = self._manager.createCamera(cam_id)
        self._cam.initialize()

        # Pick use case
        uc_list = [self._cam.getUseCases()[i]
                   for i in range(self._cam.getUseCases().size())]
        if self._use_case not in uc_list:
            print(f"[Flexx2] '{self._use_case}' not found, using '{uc_list[0]}'")
            self._use_case = uc_list[0]

        self._cam.setUseCase(self._use_case)
        self._cam.registerDataListener(self._listener)
        self._cam.startCapture()

        print(f"[Flexx2] Connected  id={self._cam.getId()}  "
              f"use_case={self._use_case}  fps={self._cam.getFrameRate()}")

    def stop(self):
        """Stop streaming.  The royale objects stay alive until GC."""
        if self._cam is not None:
            self._cam.stopCapture()
            # Let the capture thread finish any in-flight callback.
            time.sleep(0.3)
        print("[Flexx2] Stopped.")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def get_frame(self, timeout: float = 3.0) -> tuple:
        """Return one frame as (xyz, gray, conf).

        xyz  : (H, W, 3) float32  — x, y, z in metres
        gray : (H, W)    uint16   — IR amplitude image
        conf : (H, W)    uint8    — depth confidence
        """
        frame = self._listener.get_frame(timeout)
        return frame["xyz"], frame["gray"], frame["conf"]

    def get_point_cloud(self, timeout: float = 3.0) -> np.ndarray:
        """Return (N, 3) float32 array of valid 3-D points (metres)."""
        xyz, _, _ = self.get_frame(timeout)
        pts = xyz.reshape(-1, 3)
        return pts[np.any(pts != 0, axis=1)]

    def get_depth_image(self, timeout: float = 3.0) -> np.ndarray:
        """Return (H, W) float32 depth image in metres."""
        xyz, _, _ = self.get_frame(timeout)
        return xyz[:, :, 2]

    def get_ir_image(self, timeout: float = 3.0) -> np.ndarray:
        """Return (H, W) uint16 IR amplitude image."""
        _, gray, _ = self.get_frame(timeout)
        return gray

    def get_pointcloud_in_world(self, timeout: float = 3.0) -> np.ndarray:
        """Capture a point cloud and transform it into the world frame.

        Uses ``self.T_cam_world`` (set at init from calibration) to map
        every valid 3-D point from the camera frame into the world frame.

        Returns
        -------
        pts_world : (N, 3) float64 array of valid points in world coordinates (metres)
        """
        xyz, _, _ = self.get_frame(timeout)

        # Flatten and remove invalid (zero) points
        pts_cam = xyz.reshape(-1, 3).astype(np.float64)
        valid = np.any(pts_cam != 0, axis=1)
        pts_cam = pts_cam[valid]

        # T_cam_world maps  world → camera, so its inverse maps camera → world
        # T_world_cam = np.linalg.inv(self.T_cam_world)
        T_world_cam = self.T_world_cam.copy()
        R = T_world_cam[:3, :3]
        t = T_world_cam[:3,  3]

        pts_world = (R @ pts_cam.T).T + t
        return pts_world*1e3

    def get_pointcloud_in_world_filtered(
        self,
        x_range: tuple | None = None,
        y_range: tuple | None = None,
        z_range: tuple | None = None,
        timeout: float = 3.0,
    ) -> np.ndarray:
        """Capture a point cloud in world frame and filter by XYZ ranges.

        Parameters
        ----------
        x_range : (min, max) in metres, or None to skip filtering on X
        y_range : (min, max) in metres, or None to skip filtering on Y
        z_range : (min, max) in metres, or None to skip filtering on Z
        timeout : frame capture timeout in seconds

        Returns
        -------
        pts : (N, 3) float64 — filtered points in world frame (metres)

        Example
        -------
        ::

            pts = cam.get_pointcloud_in_world_filtered(
                x_range = (0.0, 0.5),
                y_range = (-0.1, 0.1),
                z_range = (0.0, 0.3),
            )
        """
        pts = self.get_pointcloud_in_world(timeout)

        if x_range is not None:
            pts = pts[(pts[:, 0] >= x_range[0]) & (pts[:, 0] <= x_range[1])]
        if y_range is not None:
            pts = pts[(pts[:, 1] >= y_range[0]) & (pts[:, 1] <= y_range[1])]
        if z_range is not None:
            pts = pts[(pts[:, 2] >= z_range[0]) & (pts[:, 2] <= z_range[1])]

        return pts

    def visualize_pointcloud(
        self,
        pts: np.ndarray,
        color_axis: int = 2,
        window_name: str = "Point cloud (world frame)",
    ):
        """Visualize a point cloud with Open3D, coloured by one world axis.

        Parameters
        ----------
        pts        : (N, 3) array in world frame
        color_axis : axis used for colouring — 0=X, 1=Y, 2=Z (default)
        window_name: Open3D window title
        """
        import open3d as o3d
        import matplotlib.pyplot as plt

        if len(pts) == 0:
            print("[Flexx2] visualize_pointcloud: no points to display.")
            return

        vals  = pts[:, color_axis]
        v_min, v_max = vals.min(), vals.max()
        norm  = (vals - v_min) / (v_max - v_min + 1e-9)
        rgb   = plt.cm.viridis(norm)[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        # World-frame axis indicator at origin
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.05, origin=[0, 0, 0]
        )
        print("x-red, y-green, z-blue")
        print(f"[Flexx2] showing {len(pts)} points — close window to continue.")
        o3d.visualization.draw_geometries(
            [pcd, axis],
            window_name=window_name,
            width=960, height=720,
        )

    # ------------------------------------------------------------------
    # AprilTag detection
    # ------------------------------------------------------------------

    def detect_tags(
        self,
        tag_size_m: float = 0.0225,
        tag_family: str = "tag16h5",
        show: bool = True,
        timeout: float = 3.0,
    ) -> tuple:
        """Capture one IR frame, detect AprilTags, print 3-D positions, show result.

        When ``tag_size_m`` is provided the detector solves the full 6-DOF pose
        using the camera intrinsics (more accurate than a raw point-cloud lookup).
        The tag centre translation in the camera depth frame is printed and drawn
        on the image.

        Camera depth-frame convention (axis legend shown on image):
          X  →  right   (col increases)
          Y  ↓  down    (row increases)
          Z  ⊙  into scene (depth / away from camera)

        Parameters
        ----------
        tag_size_m : physical side length of the tag's black border in metres.
                     Default 0.0225 (2.25 cm).
        tag_family : AprilTag family string.  Default ``'tag16h5'``.
        show       : display the annotated image.  Default ``True``.
        timeout    : seconds to wait for a frame.  Default 3.0.

        Returns
        -------
        detections : list of pupil_apriltags.Detection
                     Each detection has ``.pose_t`` (3,1) and ``.pose_R`` (3,3)
                     when tag_size_m is given.
        annotated  : (H, W, 3) uint8 BGR image with tags drawn on
        """
        from pupil_apriltags import Detector

        xyz, gray, _ = self.get_frame(timeout)
        H, W = gray.shape

        # Normalise to uint8
        gray_f  = gray.astype(np.float32)
        gray_u8 = ((gray_f - gray_f.min()) /
                   (gray_f.ptp() + 1e-9) * 255).astype(np.uint8)

        # Get camera intrinsics for pose estimation
        camera_params = None
        try:
            lp = self._cam.getLensParameters()
            def _lp(key, fallback=None):
                try: return float(lp[key])
                except Exception: return fallback
            fx = _lp("fx") or _lp("focalLengthX")
            fy = _lp("fy") or _lp("focalLengthY")
            cx = _lp("cx") or _lp("principalPointX")
            cy_p = _lp("cy") or _lp("principalPointY")
            if all(v is not None for v in (fx, fy, cx, cy_p)):
                camera_params = (fx, fy, cx, cy_p)
        except Exception:
            pass

        detector = Detector(families=tag_family, nthreads=2,
                            quad_decimate=1.0, refine_edges=True)

        if camera_params is not None:
            detections = detector.detect(gray_u8, estimate_tag_pose=True,
                                         camera_params=camera_params,
                                         tag_size=tag_size_m)
        else:
            detections = detector.detect(gray_u8)

        print(f"[Flexx2] detect_tags: found {len(detections)} tag(s) "
              f"{[d.tag_id for d in detections]}")

        annotated     = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
        palette       = [
            (0, 255, 0), (0, 128, 255), (0, 0, 255), (255, 0, 0),
            (255, 0, 255), (0, 255, 255), (255, 255, 0), (128, 0, 255),
        ]
        corner_labels = ["BL", "BR", "TR", "TL"]

        # Collect sidebar text rows for all tags
        sidebar_rows = []   # list of (text, colour)

        for i, det in enumerate(detections):
            col     = palette[i % len(palette)]
            corners = det.corners.astype(np.int32)

            # Tag outline
            cv2.polylines(annotated, [corners.reshape(-1, 1, 2)],
                          isClosed=True, color=col, thickness=2)

            # Corner dots + labels
            for j, (cx_, cy_) in enumerate(corners):
                cv2.circle(annotated, (cx_, cy_), 4, col, -1)
                cv2.putText(annotated, corner_labels[j],
                            (cx_ + 4, cy_ - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1,
                            cv2.LINE_AA)

            # Cross-hair at centre
            cx_px, cy_px = det.center.astype(int)
            origin_px = (cx_px, cy_px)
            cv2.drawMarker(annotated, origin_px, col,
                           cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)

            # ----------------------------------------------------------
            # Tag frame axes drawn in pixel space.
            # pupil_apriltags corner order: BL=0, BR=1, TR=2, TL=3
            #   Tag X  →  right  (BL→BR direction)
            #   Tag Y  ↑  up     (BL→TL direction; opposes image Y-down)
            #   Tag Z  ⊙  toward camera (out of tag face)
            # We derive the pixel directions from the corner geometry,
            # so no intrinsics are required.
            # ----------------------------------------------------------
            fc = det.corners.astype(np.float32)  # (4,2) col,row
            # X: average of (BR-BL) and (TR-TL)
            tag_x_px = ((fc[1] - fc[0]) + (fc[2] - fc[3])) / 2.0
            # Y: average of (TL-BL) and (TR-BR)  — points upward in image
            tag_y_px = ((fc[3] - fc[0]) + (fc[2] - fc[1])) / 2.0

            # Scale to fixed display length
            axis_len = 40   # pixels
            def _tip(direction):
                n = np.linalg.norm(direction)
                if n < 1e-6:
                    return origin_px
                d = direction / n * axis_len
                return (int(cx_px + d[0]), int(cy_px + d[1]))

            tip_x = _tip(tag_x_px)
            tip_y = _tip(tag_y_px)

            # X — red
            cv2.arrowedLine(annotated, origin_px, tip_x,
                            (0, 0, 220), 2, cv2.LINE_AA, tipLength=0.25)
            cv2.putText(annotated, "Xt", (tip_x[0] + 3, tip_x[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1,
                        cv2.LINE_AA)
            # Y — green
            cv2.arrowedLine(annotated, origin_px, tip_y,
                            (0, 200, 0), 2, cv2.LINE_AA, tipLength=0.25)
            cv2.putText(annotated, "Yt", (tip_y[0] + 3, tip_y[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1,
                        cv2.LINE_AA)
            # Z — blue filled circle (pointing toward camera = out of screen)
            cv2.circle(annotated, origin_px, 7, (220, 100, 0), -1)
            cv2.circle(annotated, origin_px, 7, (255, 255, 255), 1)
            cv2.putText(annotated, "Zt", (cx_px + 10, cy_px - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 100, 0), 1,
                        cv2.LINE_AA)

            # Translation
            if camera_params is not None and det.pose_t is not None:
                tx, ty, tz = det.pose_t.flatten()
                source = "pose"
            else:
                row    = int(np.clip(cy_px, 0, H - 1))
                col_px = int(np.clip(cx_px, 0, W - 1))
                tx, ty, tz = xyz[row, col_px]
                source = "pcl"

            print(f"  tag id={det.tag_id}  centre_px=({cx_px},{cy_px})  "
                  f"translation [{source}] in camera frame: "
                  f"X={tx:+.4f}m  Y={ty:+.4f}m  Z={tz:.4f}m")

            # Accumulate sidebar entries for this tag
            sidebar_rows.append((f"--- tag id={det.tag_id} [{source}] ---", col))
            sidebar_rows.append((f"  X = {tx:+.4f} m", col))
            sidebar_rows.append((f"  Y = {ty:+.4f} m", col))
            sidebar_rows.append((f"  Z = {tz:.4f} m", col))
            sidebar_rows.append(("", (0, 0, 0)))   # blank separator

        # ---------------------------------------------------------------
        # Camera depth-frame axis legend (top-left of image)
        # ---------------------------------------------------------------
        ox, oy = 50, 50
        ax_len = 35
        cv2.arrowedLine(annotated, (ox, oy), (ox + ax_len, oy),
                        (0, 0, 220), 2, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(annotated, "X", (ox + ax_len + 3, oy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1, cv2.LINE_AA)
        cv2.arrowedLine(annotated, (ox, oy), (ox, oy + ax_len),
                        (0, 200, 0), 2, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(annotated, "Y", (ox - 14, oy + ax_len + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
        cv2.circle(annotated, (ox, oy), 6, (220, 100, 0), -1)
        cv2.putText(annotated, "Z", (ox - 16, oy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 100, 0), 1, cv2.LINE_AA)
        cv2.putText(annotated, "cam frame", (ox - 42, oy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1,
                    cv2.LINE_AA)

        # Status bar at top of image
        status = f"{tag_family} | {len(detections)} tag(s) found"
        cv2.putText(annotated, status, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
                    cv2.LINE_AA)

        # ---------------------------------------------------------------
        # Build sidebar panel and stitch to the right of the image
        # ---------------------------------------------------------------
        sidebar_w  = 240
        line_h     = 22
        pad        = 10
        sidebar    = np.zeros((H, sidebar_w, 3), dtype=np.uint8)

        # Title
        cv2.putText(sidebar, "Camera frame:", (pad, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
                    cv2.LINE_AA)
        cv2.putText(sidebar, "  X right  Y down  Z depth", (pad, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1,
                    cv2.LINE_AA)
        cv2.line(sidebar, (pad, 44), (sidebar_w - pad, 44),
                 (80, 80, 80), 1)

        y = 44 + line_h
        for text, col in sidebar_rows:
            if y > H - line_h:
                break
            cv2.putText(sidebar, text, (pad, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            y += line_h

        canvas = np.hstack([annotated, sidebar])

        if show:
            win = "AprilTag detection (press any key to close)"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, W + sidebar_w, H)
            cv2.imshow(win, canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return detections, canvas

    # ------------------------------------------------------------------
    # Extrinsic calibration
    # ------------------------------------------------------------------

    def calibrate_extrinsics(
        self,
        tag_size_m: float = 0.0225,
        tag_family: str = "tag16h5",
        world_T_tag: np.ndarray | None = None,
    ) -> np.ndarray:
        """Estimate camera extrinsics from an AprilTag.

        Reads 3-D corner positions directly from the point cloud (no
        intrinsics needed) then solves for the rigid transform via SVD.

        Parameters
        ----------
        tag_size_m  : black-border side length in metres (e.g. 0.05)
        tag_family  : AprilTag family string
        world_T_tag : (4,4) tag pose in world frame.
                      If given returns T_world_cam, else returns T_tag_cam.

        Returns
        -------
        T : (4, 4) float64 homogeneous transform
        """
        xyz, gray, _ = self.get_frame()
        H, W = gray.shape

        from pupil_apriltags import Detector
        gray_f  = gray.astype(np.float32)
        gray_u8 = ((gray_f - gray_f.min()) /
                   (gray_f.ptp() + 1e-9) * 255).astype(np.uint8)
        detector   = Detector(families=tag_family, nthreads=2,
                              quad_decimate=1.0, refine_edges=True)
        detections = detector.detect(gray_u8)

        if not detections:
            raise RuntimeError("No AprilTag detected.")
        det = detections[0]

        # Use detector corners directly (no cv2.cornerSubPix — causes heap
        # corruption when corners fall near image edges on small ToF images)
        corners_px = det.corners  # (4, 2) float32, col then row

        # Lift corners into 3-D from the point cloud
        pts_cam = np.zeros((4, 3), dtype=np.float64)
        for i, (col, row) in enumerate(corners_px):
            r = int(np.clip(round(row), 0, H - 1))
            c = int(np.clip(round(col), 0, W - 1))
            pts_cam[i] = xyz[r, c].astype(np.float64)

        # Tag-frame corners: BL, BR, TR, TL
        s = tag_size_m / 2.0
        pts_tag = np.array([[-s,-s,0], [s,-s,0], [s,s,0], [-s,s,0]],
                           dtype=np.float64)

        # Horn's SVD method
        mu_c, mu_t = pts_cam.mean(0), pts_tag.mean(0)
        U, _, Vt = np.linalg.svd((pts_cam - mu_c).T @ (pts_tag - mu_t))
        D = np.diag([1., 1., np.linalg.det(U @ Vt)])
        R = U @ D @ Vt
        t = mu_c - R @ mu_t

        T_cam_tag = np.eye(4); T_cam_tag[:3,:3] = R; T_cam_tag[:3,3] = t
        T_tag_cam = np.linalg.inv(T_cam_tag)

        return world_T_tag @ T_tag_cam if world_T_tag is not None else T_tag_cam

    def calibrate_extrinsics_from_tag_position(
        self,
        tag_pos_world: np.ndarray,
        tag_size_m: float = 0.0225,
        tag_family: str = "tag16h5",
    ) -> np.ndarray:
        """Calibrate camera extrinsics given the tag's position in world frame.

        Tag-to-world axis mapping (hardcoded for the current setup):
          Tag X  →  World Z
          Tag Y  →  World X
          Tag Z  →  World -Y

        This gives the fixed rotation:

            R_world_tag = [[0,  1,  0],   # world X = tag Y
                           [0,  0, -1],   # world Y = -tag Z
                           [1,  0,  0]]   # world Z = tag X

        Parameters
        ----------
        tag_pos_world : (3,) array — [x, y, z] position of the tag origin
                        in the world / robot-base frame (metres).
        tag_size_m    : tag black-border side length in metres. Default 0.0225.
        tag_family    : AprilTag family. Default 'tag16h5'.

        Returns
        -------
        T_world_cam : (4, 4) float64
            Homogeneous transform of the camera origin in the world frame.
            The upper-left 3x3 is the rotation, the last column is the
            camera origin expressed in world coordinates.

        Example
        -------
        ::

            cam = Flexx2Driver()
            cam.connect()

            # Tag is placed at world position (0.5, 0.0, 0.3) metres
            T_world_cam = cam.calibrate_extrinsics_from_tag_position(
                tag_pos_world = np.array([0.5, 0.0, 0.3])
            )
            print("Camera origin in world frame (m):", T_world_cam[:3, 3])
            cam.stop()
        """
        # Fixed rotation: tag frame → world frame
        #   world X = tag Y  →  R col 0 = [0, 1, 0] in tag coords
        #   world Y = -tag Z →  R col 1 = [0, 0, -1]
        #   world Z = tag X  →  R col 2 = [1, 0, 0]
        R_world_tag = np.array([
            [0,  1,  0],
            [0,  0, -1],
            [1,  0,  0],
        ], dtype=np.float64)

        world_T_tag = np.eye(4)
        world_T_tag[:3, :3] = R_world_tag
        world_T_tag[:3,  3] = np.asarray(tag_pos_world, dtype=np.float64)
        
        T_world_cam = self.calibrate_extrinsics(
            tag_size_m  = tag_size_m,
            tag_family  = tag_family,
            world_T_tag = world_T_tag,
        )
        print(1)
        t = T_world_cam[:3, 3]
        R = T_world_cam[:3, :3]
        print("\nT_world_cam:")
        for i in range(3):
            for j in range(3):
                print(f"{R[i, j]:+.4f} ", end="")
            print()

        print(f"Camera origin in world frame (m): "
              f"X={t[0]:+.4f}  Y={t[1]:+.4f}  Z={t[2]:+.4f}")
        return T_world_cam

    # ------------------------------------------------------------------
    # Camera info
    # ------------------------------------------------------------------

    def get_camera_info(self) -> dict:
        uc = self._cam.getUseCases()
        info = {
            "id":           self._cam.getId(),
            "name":         self._cam.getCameraName(),
            "width":        self._cam.getMaxSensorWidth(),
            "height":       self._cam.getMaxSensorHeight(),
            "frame_rate":   self._cam.getFrameRate(),
            "use_cases":    [uc[i] for i in range(uc.size())],
            "current_use_case": self._cam.getCurrentUseCase(),
        }
        try:
            lp = self._cam.getLensParameters()
            info["lens_parameters"] = {u: lp[u] for u in lp}
        except Exception:
            info["lens_parameters"] = {}
        return info

    def print_camera_info(self):
        info = self.get_camera_info()
        print("=" * 40)
        for k, v in info.items():
            print(f"  {k:<22}: {v}")
        print("=" * 40)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def list_cameras(access_code: str = "") -> list:
        mgr = roypy.CameraManager(access_code)
        cl  = mgr.getConnectedCameraList()
        return [cl[i] for i in range(cl.size())]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cam = Flexx2Driver()
    cam.connect()
    # cam.print_camera_info()

    cam.detect_tags()
    # T_world_cam = cam.calibrate_extrinsics_from_tag_position(
    #     tag_pos_world = np.array([25.18, -12.64, 119])*1e-3  # metres
    # )
    cam.stop()
    exit(0)
    print("T_world_cam:")
    print("type:", type(T_world_cam))
    print(T_world_cam[0,3])
    print(T_world_cam[1,3])
    print(T_world_cam[2,3])
    # Filter to a region of interest in world frame, then visualize
    pts = cam.get_pointcloud_in_world_filtered(
        x_range = (-1,  1.0),
        y_range = (-1.0, 0.4),
        z_range = (0.0,  0.3),
    )
    pts = cam.get_point_cloud()
    print("minz:", np.min(pts[:, 2]))
    print("maxz:", np.max(pts[:, 2]))

    # pts = cam.get_pointcloud_in_world_filtered()
    cam.visualize_pointcloud(pts)  # coloured by Z by default

    # Colour by X instead
    # cam.visualize_pointcloud(pts, color_axis=0)

    cam.stop()