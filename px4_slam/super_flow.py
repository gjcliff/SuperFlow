import time
from typing import Any, cast

import cv2
import numpy as np
import rclpy
import rerun as rr
import torch
from lightglue import SuperPoint
from px4_msgs.msg import SensorGps, VehicleOdometry
from px4_slam_interfaces.msg import Keyframe as KeyframeMsg
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial import KDTree
from sensor_msgs.msg import Image

from px4_slam.data import IDGenerator, Keyframe

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

        rr.init("superflow", recording_id=recording_id)
        rr.spawn()
        rr.log("world", rr.ViewCoordinates.FRD, static=True)

        self._keyframe_pub: rclpy.node.Publisher = self.create_publisher(
            KeyframeMsg, "superflow/keyframe", qos_profile_sensor_data
        )
        self._image_sub: rclpy.node.Subscription = self.create_subscription(
            Image, "camera/image_raw", self.image_callback, qos_profile_sensor_data
        )
        self._odometry_sub: rclpy.node.Subscription = self.create_subscription(
            VehicleOdometry,
            "fmu/out/vehicle_odometry",
            self.odometry_callback,
            qos_profile=qos_profile_sensor_data,
        )
        # self._gps_sub: rclpy.node.Subscription = self.create_subscription(
        #     SensorGps,
        #     "fmu/out/sensor_gps",
        #     self.gps_callback,
        #     qos_profile=qos_profile_sensor_data,
        # )

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
        self.min_track_length: int = 30
        self.kp_match_thresh: float = 0.8
        self.max_tracks = 50_000

        # track state
        self.prev_gray: np.ndarray | None = None
        self.prev_pts: np.ndarray | None = None  # this will go
        self.prev_track_ids: np.ndarray | None = None
        self.mask: np.ndarray | None = None  # = np.zeros(img_shape, dtype=np.uint8)

        self.kfs: dict[int, Keyframe] = {}
        self.kdtree: KDTree | None = None
        self.track_counts: np.ndarray = np.empty(0, dtype=np.uint32)
        self.min_kfs: int = min_kfs

        self.latest_odom_msg: VehicleOdometry | None = None
        self.latest_gps_msg: SensorGps | None = None
        self.ref_sin_lat: float | None = None
        self.ref_cos_lat: float | None = None
        self.ref_lat: float | None = None
        self.ref_lon: float | None = None
        self.ref_alt: float | None = None

    def gray_img_to_tensor(self, img: np.ndarray):
        tensor = torch.from_numpy(img).float() / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        return tensor.cuda()

    def rgb_img_to_tensor(self, img: np.ndarray):
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

    def ros_image_to_rgb(self, msg: Image) -> np.ndarray:
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
        gpu_img = self.gray_img_to_tensor(img)
        feats = self.extractor.extract(gpu_img)
        kps = feats["keypoints"][0].cpu().numpy()  # (n, 2)
        desc = feats["descriptors"][0].cpu().numpy()  # (n, 256)

        return kps.reshape(-1, 2).astype(np.float32), desc

    def draw_track_ids(self, img: np.ndarray, pts: np.ndarray, track_ids: np.ndarray):
        for pt, tid in zip(pts, track_ids):
            x, y = int(pt[0]), int(pt[1])
            count = self.track_counts[tid]
            cv2.putText(
                img,
                f"{tid}, {count}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 255, 0),
                1,
            )

    def rebuild_kdtree(self):
        positions = np.array([kf.position for kf in self.kfs.values()])
        self.kdtree = KDTree(positions)

    def create_kf(
        self,
        new_pts: np.ndarray,
        new_descs: np.ndarray,
        img_size: tuple[int, int],
        track_ids: np.ndarray | None = None,
    ) -> Keyframe | None:
        # am i cheating?
        # this could also come from gps and attitude
        if self.latest_odom_msg is None:
            return None

        # tracks handle creating and incrementing their own ids

        # keyframes also handle creating and incrementing their own ids
        track_ids = (
            np.zeros(new_pts.shape[0], dtype=np.uint32)
            if track_ids is None
            else track_ids
        )
        kf = Keyframe(
            kps=new_pts,
            desc=new_descs,
            track_ids=track_ids,
            position=self.latest_odom_msg.position,
            q=self.latest_odom_msg.q,
            img_size=img_size,
            log=True,
        )
        self.get_logger().info(
            f"new kf with {kf.kps.shape} points, {kf.desc.shape} desc, {kf.track_ids.shape} tracks"
        )

        self.publish_kf(kf)

        return kf

    def get_recent_keyframes(self, n: int) -> list[Keyframe]:
        """Return the last n kfs in a list"""
        keys = list(self.kfs.keys())
        recent_keys = keys[-n:]
        return [self.kfs[k] for k in recent_keys]

    def get_latest_keyframe(self) -> Keyframe | None:
        """Get the most recent kf"""
        if self.kfs:
            return self.get_recent_keyframes(n=1)[0]
        return None

    def associate_tracks(self, new_kf: Keyframe):
        """Update track_ids in the new kf if they already exist

        this is mainly a front-end only function, we just want it to track points from
        one kf to another. loop closure if for older kfs
        """
        kf = self.get_latest_keyframe()
        if kf is None:
            return
        scores = new_kf.desc @ kf.desc.T
        matches = np.argmax(scores, axis=1)

        best_scores = scores[np.arange(len(matches)), matches]
        valid = best_scores > self.kp_match_thresh
        self.get_logger().debug(f"num valid: {len(valid)}")
        matched_track_ids = kf.track_ids[matches]
        new_kf.track_ids[valid] = matched_track_ids[valid]

    def register_keyframe(self, new_kf: Keyframe):
        """Add new tracks, update old ones, and add the kf to the kf dict"""
        # a track's track_id is its index in track_counts
        # if a track_id is a larger number than the length of track_counts, that must
        # mean that it's a new id
        existing_ids = new_kf.track_ids > 0
        brand_new_ids = ~existing_ids
        n_new = brand_new_ids.sum()
        new_ids = IDGenerator.next_batch(n=n_new)
        new_kf.track_ids[brand_new_ids] = new_ids

        self.track_counts = np.concat(
            [self.track_counts, np.ones(n_new, dtype=np.uint32)]
        )

        self.track_counts[new_kf.track_ids[existing_ids]] += 1
        self.kfs[new_kf.kf_id] = new_kf
        self.get_logger().info(
            f"keyframe {new_kf.kf_id} registered, {n_new} new tracks"
        )
        self.rebuild_kdtree()

    def redetect_and_merge(self, img: np.ndarray) -> Keyframe | None:
        if self.latest_odom_msg is None:
            return

        new_pts, new_descs = self.detect_with_superpoint(img)

        new_kf = self.create_kf(
            new_pts=new_pts, new_descs=new_descs, img_size=img.shape
        )
        if new_kf is None:
            return

        if new_kf.log:
            new_kf.log_keyframe_img(img)

        self.associate_tracks(new_kf)
        self.register_keyframe(new_kf)
        self.get_logger().info(f"kfs: {len(self.kfs)}")

        return new_kf

    def check_loop_closure(self, new_kf: Keyframe):
        candidates = self.get_loop_closure_candidates(new_kf)

    def get_loop_closure_candidates(self, new_kf: Keyframe) -> list[Keyframe]:
        closest = self.get_closest_keyframes(new_kf.position, k=self.min_kfs)
        candidates = []

        for kf in closest:
            if abs(kf.kf_id - new_kf.kf_id) < 2:  # hmm
                continue
            dot = np.abs(np.dot(new_kf.q, kf.q))
            if dot > 0.9:
                candidates.append(kf)

        return candidates

    def get_closest_keyframes(self, position: np.ndarray, k: int = 5) -> list[Keyframe]:
        if self.kdtree is None or len(self.kfs) < k:
            return []

        _, indices = self.kdtree.query(position, k=k)
        kf_ids = list(self.kfs.keys())
        return [self.kfs[kf_ids[i]] for i in indices]

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
        gray = self.ros_image_to_gray(msg)

        # redetect with superpoint periodically or on first frame
        if self.prev_pts is None or self.prev_pts.shape[0] < 30:
            kf = self.redetect_and_merge(gray)
            self.prev_pts = kf.kps if kf else None
            self.prev_track_ids = kf.track_ids if kf else None
            self.prev_gray = gray
            return

        pretty_img = self.ros_image_to_rgb(msg)

        # track with lk optical flow
        # had an ai help me with type casting because i like knowing types
        result = cast(
            tuple[np.ndarray, np.ndarray, np.ndarray] | None,
            cv2.calcOpticalFlowPyrLK(  # ty: ignore
                self.prev_gray, gray, self.prev_pts, None, **self.lk_params
            ),
        )
        curr_pts: np.ndarray | None = result[0] if result is not None else None
        status: np.ndarray | None = result[1] if result is not None else None
        # err: np.ndarray | None = result[2] if result is not None else None

        if curr_pts is None or status is None:
            self.get_logger().warn("optical flow failed, triggering redetection")
            self.prev_pts = None  # force redetection next frame
            return

        # ravel is like flatten, but tries to return a view and not a copy
        good_mask = np.array(
            status.ravel() == 1, dtype=bool
        )  # == 1 to create boolean mask
        pts1 = curr_pts[good_mask].reshape(-1, 2)
        if pts1.shape[0] == 0:
            self.prev_pts = None
            self.prev_gray = gray
            return

        # self.log_matches("matches", img, pts1)

        pts0 = self.prev_pts[good_mask]
        track_ids = (
            self.prev_track_ids[good_mask] if self.prev_track_ids is not None else None
        )

        try:
            if track_ids is not None:
                self.track_counts[track_ids] += 1
        except IndexError:
            breakpoint()

        output = self.draw_mask(pretty_img, pts1, pts0, track_ids)
        last_kf = self.get_latest_keyframe()
        if (
            last_kf is not None
            and track_ids is not None
            and self.should_add_kf(last_kf)
        ):
            mask, descs = last_kf.get_descs_for_tracks(track_ids)
            new_kf = self.create_kf(
                new_pts=pts1[mask],
                new_descs=descs,
                img_size=gray.shape,
                track_ids=track_ids[mask],
            )
            if new_kf is not None:
                self.add_kf(new_kf)
                self.log_kf_img(
                    output,
                    pts1,
                    pts0,
                    track_ids,
                    new_kf.kf_id,
                )

        self.prev_pts = pts1
        self.prev_gray = gray
        self.prev_track_ids = track_ids

        # TODO: log the points

    def add_kf(self, kf: Keyframe | None):
        if kf is None:
            return

        self.kfs[kf.kf_id] = kf
        self.rebuild_kdtree()

    def publish_kf(self, kf: Keyframe):
        self._keyframe_pub.publish(kf.to_ros_msg(clock=self.get_clock()))

    def should_add_kf(self, last_kf: Keyframe) -> bool:
        if self.latest_odom_msg is not None:
            if last_kf.is_close(self.latest_odom_msg.position, self.latest_odom_msg.q):
                return False
        return True

    def draw_mask(
        self,
        img: np.ndarray,
        pts1: np.ndarray,
        pts0: np.ndarray,
        track_ids: np.ndarray,
        *,
        log: bool = True,
    ):
        if self.mask is None:
            self.mask = np.zeros(img.shape, dtype=np.uint8)

        self.mask = (self.mask * 0.95).astype(np.uint8)

        for new, old in zip(pts1, pts0):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            cv2.line(self.mask, (a, b), (c, d), (0, 255, 0), 2)

        output = cv2.add(img, self.mask)
        self.draw_track_ids(output, pts1, track_ids)
        if log:
            rr.log("world/camera/flow", rr.Image(output), static=True)

        return output

    def log_kf_img(
        self,
        img: np.ndarray,
        pts1: np.ndarray,
        pts0: np.ndarray,
        track_ids: np.ndarray,
        kf_id: int,
    ):
        pretty = self.draw_mask(img, pts1, pts0, track_ids, log=False)
        rr.log(f"world/camera/keyframes/{kf_id}/pinhole", rr.Image(pretty))

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
