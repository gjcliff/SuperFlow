import time
from typing import Any, cast

import cv2
import gtsam
import numpy as np
import rclpy
import rerun as rr
import torch
from lightglue import SuperPoint
from px4_msgs.msg import SensorGps, VehicleOdometry
from px4_slam_interfaces.msg import Keyframe as KeyframeMsg
from px4_slam_interfaces.msg import LoopClosure as LoopClosureMsg
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial import KDTree
from sensor_msgs.msg import CameraInfo, Image

import px4_slam.logging as lg
from px4_slam.data import IDGenerator, Keyframe, LoopClosure

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")


class SuperFlowState:
    def __init__(self):
        self.prev_pts: np.ndarray | None = None  # this will go
        self.prev_gray: np.ndarray | None = None
        self.prev_track_ids: np.ndarray | None = None
        self.mask: np.ndarray | None = None  # = np.zeros(img_shape, dtype=np.uint8)

    def reset(self):
        self.prev_pts = None
        self.prev_gray = None
        self.prev_track_ids = None

    def update(self, pts, gray, track_ids):
        self.prev_pts = pts
        self.prev_gray = gray
        self.prev_track_ids = track_ids

    @property
    def needs_redetect(self) -> bool:
        return self.prev_pts is None or self.prev_pts.shape[0] < 100


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
        self._lc_pub: rclpy.node.Publisher = self.create_publisher(
            LoopClosureMsg, "superflow/loop_closure", qos_profile_sensor_data
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
        self._camera_info_sub: rclpy.node.Subscription | None = (
            self.create_subscription(
                CameraInfo,
                "camera/camera_info",
                self.camera_info_callback,
                qos_profile=qos_profile_sensor_data,
            )
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
        self.state = SuperFlowState()

        self.kfs: dict[int, Keyframe] = {}
        self.lcs: dict[int, LoopClosure]
        self.kdtree: KDTree | None = None
        self.track_counts: np.ndarray = np.empty(1, dtype=np.uint32)
        self.min_kfs: int = min_kfs
        self.min_loop_closure_matches = 50
        self.lc_done = False

        self.K: gtsam.Cal3_S2 | None = None
        self.body_T_cam = np.array(
            [
                [0, 0, 1, 0.12],
                [1, 0, 0, 0.03],
                [0, 1, 0, 0.242],
                [0, 0, 0, 1],
            ]
        )

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
        kf = self.get_latest_keyframe()
        if kf is None:
            return
        scores = new_kf.desc @ kf.desc.T  # (n_new, n_old)
        matches = np.argmax(scores, axis=1)
        best_scores = scores[np.arange(len(matches)), matches]
        valid = best_scores > self.kp_match_thresh

        # for each old track, only keep the new point with the highest score
        valid_indices = np.where(valid)[0]

        # among duplicates keep the highest scoring one
        final_mask = np.zeros(len(new_kf.track_ids), dtype=bool)
        assigned_old = {}
        for idx in valid_indices[np.argsort(-best_scores[valid])]:
            old_id = matches[idx]
            if old_id not in assigned_old:
                assigned_old[old_id] = idx
                final_mask[idx] = True

        new_kf.track_ids[final_mask] = kf.track_ids[matches[final_mask]]

    def register_keyframe(self, new_kf: Keyframe):
        """Add new tracks, update old ones, and add the kf to the kf dict"""
        # a track's track_id is its index in track_counts
        # if a track_id is a larger number than the length of track_counts, that must
        # mean that it's a new id
        brand_new_ids = new_kf.track_ids == 0
        existing_ids = ~brand_new_ids
        n_new = int(brand_new_ids.sum())
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
        self.publish_kf(new_kf)
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

        self.associate_tracks(new_kf)
        self.register_keyframe(new_kf)

        if len(np.unique(new_kf.track_ids)) != len(new_kf.track_ids):
            breakpoint()
        self.check_loop_closure(new_kf)

        return new_kf

    def check_loop_closure(self, new_kf: Keyframe, *, log: bool = True):
        if self.lc_done:
            return

        candidates = self.get_loop_closure_candidates(new_kf)

        if log and candidates:
            lg.log_lc_candidates(new_kf, candidates)

        if self.K is None:
            return

        for kf in candidates:
            scores = new_kf.desc @ kf.desc.T
            matches = np.argmax(scores, axis=1)
            best_scores = scores[np.arange(len(matches)), matches]
            valid = best_scores > self.kp_match_thresh
            if (n_valid := int(valid.sum())) < self.min_loop_closure_matches:
                continue

            lg.log_loop_closure(new_kf, kf, n_valid)
            rel_pos, rel_q = self.get_relative_pose(new_kf, kf)
            self.lc_done = True
            self.pub_loop_closure(
                LoopClosure(
                    query_kf_id=new_kf.kf_id,
                    match_kf_id=kf.kf_id,
                    rel_pos=rel_pos,
                    rel_q=rel_q,
                    n_inliers=n_valid,
                )
            )

    def get_relative_pose(self, kf_from: Keyframe, kf_to: Keyframe):
        pose_from = gtsam.Pose3(
            gtsam.Rot3.Quaternion(*kf_from.q),
            gtsam.Point3(*kf_from.position),
        )
        pose_to = gtsam.Pose3(
            gtsam.Rot3.Quaternion(*kf_to.q),
            gtsam.Point3(*kf_to.position),
        )

        relative = pose_from.inverse().compose(pose_to)
        t = relative.translation()
        q = relative.rotation().toQuaternion()

        return np.array(t), np.array([q.w(), q.x(), q.y(), q.z()])

    def pub_loop_closure(self, lc: LoopClosure):
        msg = lc.to_ros_msg(self.get_clock())
        self._lc_pub.publish(msg)

    def get_loop_closure_candidates(self, new_kf: Keyframe) -> list[Keyframe]:
        closest = self.get_closest_keyframes(new_kf.position, k=self.min_kfs)
        candidates = []

        for kf in closest:
            if abs(kf.kf_id - new_kf.kf_id) < 20:  # hmm
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

    def increment_track_ids(self, track_ids: np.ndarray):
        self.track_counts[track_ids] += 1

    def image_callback(self, msg: Image):
        gray = self.ros_image_to_gray(msg)
        pretty_img = self.ros_image_to_rgb(msg)

        # redetect with superpoint periodically or on first frame
        if self.state.needs_redetect:
            kf = self.redetect_and_merge(gray)
            if kf is None:
                return
            self.log_kf(kf, pretty_img)
            self.state.update(pts=kf.kps, gray=gray, track_ids=kf.track_ids)
            return

        # track with lk optical flow
        # had an ai help me with type casting because i like knowing types
        result = cast(
            tuple[np.ndarray, np.ndarray, np.ndarray] | None,
            cv2.calcOpticalFlowPyrLK(  # ty: ignore
                self.state.prev_gray, gray, self.state.prev_pts, None, **self.lk_params
            ),
        )
        curr_pts: np.ndarray | None = result[0] if result is not None else None
        status: np.ndarray | None = result[1] if result is not None else None
        # err: np.ndarray | None = result[2] if result is not None else None

        if curr_pts is None or status is None:
            self.get_logger().warn("optical flow failed, triggering redetection")
            self.state.reset()
            return

        # ravel is like flatten, but tries to return a view and not a copy
        good_mask = np.array(
            status.ravel() == 1, dtype=bool
        )  # == 1 to create boolean mask
        pts1 = curr_pts[good_mask].reshape(-1, 2)
        if pts1.shape[0] == 0:
            self.state.reset()
            return

        pts0 = self.state.prev_pts[good_mask]
        track_ids = cast(
            np.ndarray,
            self.state.prev_track_ids[good_mask]
            if self.state.prev_track_ids is not None
            else None,
        )

        try:
            if track_ids is not None:
                self.increment_track_ids(track_ids)
                self.track_counts[track_ids] += 1
        except IndexError:
            breakpoint()

        _ = self.draw_mask(pretty_img, pts1, pts0, track_ids)

        self.state.update(pts=pts1, gray=gray, track_ids=track_ids)

    def add_kf(self, kf: Keyframe | None):
        if kf is None:
            return

        self.kfs[kf.kf_id] = kf
        self.rebuild_kdtree()

    def publish_kf(self, kf: Keyframe):
        self._keyframe_pub.publish(kf.to_ros_msg(clock=self.get_clock()))

    # def should_add_kf(self, last_kf: Keyframe) -> bool:
    #     return False
    #     if self.latest_odom_msg is not None:
    #         if last_kf.is_close(self.latest_odom_msg.position, self.latest_odom_msg.q):
    #             return False
    #     return True

    def draw_mask(
        self,
        img: np.ndarray,
        pts1: np.ndarray,
        pts0: np.ndarray,
        track_ids: np.ndarray,
        *,
        log: bool = True,
    ):
        if self.state.mask is None:
            self.state.mask = np.zeros(img.shape, dtype=np.uint8)

        self.state.mask = (self.state.mask * 0.95).astype(np.uint8)

        for new, old in zip(pts1, pts0):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            cv2.line(self.state.mask, (a, b), (c, d), (0, 255, 0), 2)

        output = cv2.add(img, self.state.mask)
        self.draw_track_ids(output, pts1, track_ids)
        if log:
            rr.log("flow", rr.Image(output), static=True)

        return output

    def log_kf(self, kf: Keyframe, img: np.ndarray):
        if self.K is None:
            return

        lg.log_kf(
            K=self.K,
            body_T_cam=self.body_T_cam,
            kf=kf,
            img=img,
            track_counts=self.track_counts,
        )

    def odometry_callback(self, msg: VehicleOdometry):
        self.latest_odom_msg = msg

    def gps_callback(self, msg: SensorGps):
        self.latest_gps_msg = msg

    # [fx,  s, cx]     k[0], k[1], k[2]
    # [ 0, fy, cy]  =  k[3], k[4], k[5]
    # [ 0,  0,  1]     k[6], k[7], k[8]
    def camera_info_callback(self, msg: CameraInfo):
        if self.K is None:
            self.K = gtsam.Cal3_S2(
                msg.k[0],  # fx
                msg.k[4],  # fy
                msg.k[1],  # s (skew)
                msg.k[2],  # cx
                msg.k[5],  # cy
            )

            if self._camera_info_sub is not None:
                self.destroy_subscription(self._camera_info_sub)
                self._camera_info_sub = None


def main(args=None):
    rclpy.init(args=args)
    super_flow = SuperFlow()
    rclpy.spin(super_flow)
    super_flow.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
