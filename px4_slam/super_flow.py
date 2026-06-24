import time
from typing import Any, cast

import cv2
import numpy as np
import rclpy
import rerun as rr
import torch
from geometry_msgs.msg import PoseStamped
from lightglue import SuperPoint
from px4_msgs.msg import SensorGps, VehicleOdometry
from px4_slam_interfaces.msg import LoopClosure, MatchedPoints
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image

from px4_slam.data import Keyframe, Track

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")


class SuperFlow(Node):
    def __init__(self):
        super().__init__("super_flow")

        self.declare_parameter("recording_id", str(int(time.time())))
        self.declare_parameter("min_kfs", 5)
        recording_id = (
            self.get_parameter("recording_id").get_parameter_value().string_value
        )
        min_kfs = self.get_parameter("min_kfs").get_parameter_value().integer_value

        rr.init("super_flow", recording_id=recording_id)
        rr.spawn()

        self._matched_points_pub: rclpy.node.Publisher = self.create_publisher(
            MatchedPoints, "camera/matched_points", qos_profile_sensor_data
        )
        self._loop_closure_pub: rclpy.node.Publisher = self.create_publisher(
            LoopClosure, "camera/loop_closure", qos_profile_sensor_data
        )
        self._image_sub: rclpy.node.Subscription = self.create_subscription(
            Image, "camera/image_raw", self.image_callback, qos_profile_sensor_data
        )
        self._state_estimate_sub: rclpy.node.Subscription = self.create_subscription(
            PoseStamped,
            "state_estimate/pose",
            self.pose_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self._odometry_sub: rclpy.node.Subscription = self.create_subscription(
            VehicleOdometry,
            "fmu/out/vehicle_odometry",
            self.odometry_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self._gps_sub: rclpy.node.Subscription = self.create_subscription(
            SensorGps,
            "fmu/out/sensor_gps",
            self.gps_callback,
            qos_profile=qos_profile_sensor_data,
        )

        self.latest_camera_info_msg: CameraInfo | None
        self.latest_pose: PoseStamped | None = None
        self.latest_odom_msg: VehicleOdometry | None = None
        self.latest_gps_msg: SensorGps | None = None
        self.min_separation: float = 2.0
        self.keyframe_db: list[dict] = []
        self.ref_sin_lat: float | None = None
        self.ref_cos_lat: float | None = None
        self.ref_lat: float | None = None
        self.ref_lon: float | None = None
        self.ref_alt: float | None = None

        self.last_keyframe_pose: np.ndarray | None = None
        self.keyframe_translation_threshold: float = 1.0
        self.keyframe_rotation_threshold: float = np.radians(20)

        # superpoint for detection only, no matcher needed
        self.extractor: SuperPoint = (
            SuperPoint(max_num_keypoints=128, detection_threshold=0.005, nms_radius=4)
            .eval()
            .cuda()
        )

        self.max_lost_memory: int = 30  # frames to remember lost tracks
        self.max_history_len: int = 10  # max track history for viz

        # lk optical flow params
        self.lk_params: dict[str, Any] = dict(
            winSize=(31, 31),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        self.redetect_every: int = (
            120  # redetect with superpoint every N frames, TODO: fps
        )

        # track state
        self.prev_img: np.ndarray | None = None
        self.prev_pts: np.ndarray | None = None  # this will go
        self.track_ids: list[int] = []
        self.track_lengths: dict[int, int] = {}
        # track_id -> (256,) descriptor
        self.track_descriptors: dict[int, np.ndarray] = {}
        # recently lost track_id -> descriptor
        self.lost_track_descriptors: dict[int, np.ndarray] = {}
        self.lost_track_age: dict[int, int] = {}  # track_id -> frames since lost
        # track_id -> list of (x, y)
        self.track_history: dict[int, list[tuple[int, int]]] = {}
        self.min_track_length: int = 30

        self.kfs: dict[int, Keyframe] = {}
        self.tracks: dict[int, Track] = {}
        self.min_kfs: int = min_kfs

        self.frame_count: int = 0
        self.keyframe_count: int = 0

    def numpy_img_to_tensor(self, img: np.ndarray):
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return tensor.cuda()

    def ros_image_to_tensor(self, msg: Image) -> torch.Tensor:
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        # numpy is (h)eight, (w)idth, (c)hannel, but pytorch is (c, h, w)
        # divide by 255.0 to turn 0-255 int into 0-1 float for pytorch
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if msg.encoding == "bgr8":
            tensor = tensor.flip(0)
        return tensor.cuda()

    def ros_image_to_gray(self, msg: Image) -> np.ndarray:
        # Didn't want to have dependency on cv_bridge, old numpy
        # beautiful!
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding == "bgr8":
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def ros_image_to_numpy(self, msg: Image) -> np.ndarray:
        # see previous function
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding == "bgr8":
            # new to me
            # ... -> all preceeding dimensions (h, w)
            # ::-1 -> reverse the last axis
            # RGB turns to BGR
            img = img[..., ::-1]
        return img

    def detect_with_superpoint(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gpu_img = self.numpy_img_to_tensor(img)
        feats = self.extractor.extract(gpu_img)
        kps = feats["keypoints"][0].cpu().numpy()  # (n, 2)
        desc = feats["descriptors"][0].cpu().numpy()  # (n, 256)

        return kps.reshape(-1, 1, 2).astype(np.float32), desc

    def draw_track_ids(
        self, img: np.ndarray, pts: np.ndarray, track_ids: list[int]
    ) -> np.ndarray:
        img = img.copy()
        for pt, tid in zip(pts, track_ids):
            x, y = int(pt[0]), int(pt[1])
            cv2.putText(
                img, str(tid), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1
            )
        return img

    def create_keyframe(
        self, new_pts: np.ndarray, new_descs: np.ndarray
    ) -> Keyframe | None:
        # am i cheating?
        # this could also come from gps and attitude
        if self.latest_odom_msg is None:
            return None

        # tracks handle creating and incrementing their own ids
        new_tracks = [
            Track(count=1, last_seen_frame=Keyframe.get_next_id()) for _ in new_pts
        ]
        for track in new_tracks:
            self.tracks[track.track_id] = track

        new_ids = np.array([track.track_id for track in new_tracks], dtype=np.int32)

        q = self.latest_odom_msg.q
        # keyframes also handle creating and incrementing their own ids
        kf = Keyframe(
            kps=new_pts,
            desc=new_descs,
            track_ids=new_ids,
            position=self.latest_odom_msg.position,
            q=self.latest_odom_msg.q,
            rot=Rotation(quat=[q[1], q[2], q[3], q[0]]).as_matrix(),
        )

        return kf

    def get_recent_keyframes(self, n: int) -> list[Keyframe]:
        keys = list(self.kfs.keys())
        recent_keys = keys[-n:]
        return [self.kfs[k] for k in recent_keys]

    def get_latest_keyframe(self) -> Keyframe | None:
        if self.kfs:
            return self.get_recent_keyframes(n=1)[0]
        return None

    def redetect_and_merge(self, img: np.ndarray):
        if self.latest_odom_msg is None:
            return

        # leave off for testing
        # last_kf = self.get_recent_keyframes(n=1)[0]
        # dist = np.linalg.norm(self.latest_odom_msg.position - last_kf.position)
        # if dist < 0.05:  # tune this threshold
        #     return

        new_pts, new_descs = self.detect_with_superpoint(img)

        new_kf = self.create_keyframe(new_pts=new_pts, new_descs=new_descs)
        if new_kf is None:
            return
        self.get_logger().info(
            f"redetect: first frame, Keyframe added with {len(new_pts)} kps"
        )
        self.get_logger().info(f"kfs: {len(self.kfs)}")

        recent_kfs = self.get_recent_keyframes(n=self.min_kfs)

        if len(self.kfs) < self.min_kfs:
            self.kfs[new_kf.kf_id] = new_kf

        for kf in recent_kfs:
            scores = new_kf.desc @ kf.desc.T
            matches = np.argmax(scores, axis=1)
            expected = np.arange(len(matches))
            if np.array_equal(matches, expected):
                continue

            best_scores = scores[np.arange(len(matches)), matches]
            valid = best_scores > 0.8  # how to tune this or make it auto?
            self.get_logger().debug(f"num valid: {len(valid)}")
            matched_track_ids = kf.track_ids[matches]
            # reassign track ids we've seen already
            new_kf.track_ids[valid] = matched_track_ids[valid]

            for track_id in new_kf.track_ids[valid]:
                self.tracks[track_id].count += 1
                self.tracks[track_id].last_seen_frame = new_kf.kf_id

            self.kfs[new_kf.kf_id] = new_kf

    def init_reference(self, lat_0, lon_0, alt_0):
        self.ref_alt = alt_0
        self.ref_lat = np.radians(lat_0)
        self.ref_lon = np.radians(lon_0)
        self.ref_sin_lat = np.sin(self.ref_lat)
        self.ref_cos_lat = np.cos(self.ref_lat)

    def project(self, lat, lon):
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        cos_d_lon = np.cos(lon_rad - self.ref_lon)
        arg = np.clip(
            self.ref_sin_lat * sin_lat + self.ref_cos_lat * cos_lat * cos_d_lon,
            -1.0,
            1.0,
        )
        c = np.arccos(arg)
        k = c / np.sin(c) if abs(c) > 0 else 1.0
        north = (
            k
            * (self.ref_cos_lat * sin_lat - self.ref_sin_lat * cos_lat * cos_d_lon)
            * 6371000
        )
        east = k * cos_lat * np.sin(lon_rad - self.ref_lon) * 6371000
        return north, east

    def image_callback(self, msg: Image):
        img = self.ros_image_to_numpy(msg)

        # redetect with superpoint periodically or on first frame
        if self.prev_pts is None or self.frame_count % self.redetect_every == 0:
            self.redetect_and_merge(img)
            kf = self.get_latest_keyframe()
            self.prev_pts = kf.kps if kf else None
            self.prev_img = img
            self.frame_count += 1  # ? maybe can get the boot
            return

        # track with lk optical flow
        # had an ai help me with type casting because i like knowing types
        result = cast(
            tuple[np.ndarray, np.ndarray, np.ndarray] | None,
            cv2.calcOpticalFlowPyrLK(  # ty: ignore
                self.prev_img, img, self.prev_pts, None, **self.lk_params
            ),
        )
        curr_pts: np.ndarray | None = result[0] if result is not None else None
        status: np.ndarray | None = result[1] if result is not None else None
        # err: np.ndarray | None = result[2] if result is not None else None

        if curr_pts is None or status is None:
            self.get_logger().warn("optical flow failed, triggering redetection")
            self.prev_img = None  # force redetection next frame
            self.frame_count += 1
            return

        # ravel is like flatten, but tries to return a view and not a copy
        good_mask = status.ravel() == 1  # == 1 to create boolean mask
        pts1 = curr_pts[good_mask].reshape(-1, 2)
        breakpoint()

        # TODO: log the points

    def log_kps(self, img: np.ndarray, kps: np.ndarray):
        pretty = img.copy()
        for kp in kps:
            pretty = cv2.circle(
                img=pretty, center=kp, radius=-1, color=(255, 0, 0), thickness=-1
            )

        rr.log("world/image", rr.Image(pretty))

    def pose_callback(self, msg: PoseStamped):
        self.latest_pose = msg

    def odometry_callback(self, msg: VehicleOdometry):
        self.latest_odom_msg = msg

    def gps_callback(self, msg: SensorGps):
        self.latest_gps_msg = msg


def main(args=None):
    rclpy.init(args=args)
    super_flow = SuperFlow()
    rclpy.spin(super_flow)
    super_flow.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
