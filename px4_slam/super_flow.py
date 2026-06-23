import time
from typing import Any, cast

import cv2
import numpy as np
import rclpy
import rerun as rr
import torch
from geometry_msgs.msg import Pose, PoseStamped
from lightglue import SuperPoint
from px4_msgs.msg import SensorGps, VehicleOdometry
from px4_slam_interfaces.msg import LoopClosure, MatchedPoints
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")


class SuperFlow(Node):
    def __init__(self):
        super().__init__("super_flow")

        self.declare_parameter("recording_id", str(int(time.time())))
        recording_id = (
            self.get_parameter("recording_id").get_parameter_value().string_value
        )

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
        self.latest_image_msg: Image | None
        self.latest_pose: PoseStamped | None = None
        self.latest_odom_msg: VehicleOdometry | None = None
        self.latest_gps_msg: SensorGps | None = None
        self.min_keyframe_distance: float = 5.0
        self.min_keyframes: int = 20
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
        self.prev_gray: np.ndarray | None = None
        self.prev_pts: np.ndarray | None = None  # (N, 1, 2) float32
        self.track_ids: list[int] = []
        self.track_lengths: dict[int, int] = {}
        # track_id -> (256,) descriptor
        self.track_descriptors: dict[int, np.ndarray] = {}
        # recently lost track_id -> descriptor
        self.lost_track_descriptors: dict[int, np.ndarray] = {}
        self.lost_track_age: dict[int, int] = {}  # track_id -> frames since lost
        # track_id -> list of (x, y)
        self.track_history: dict[int, list[tuple[int, int]]] = {}
        self.track_id: int = 0
        self.min_track_length: int = 30

        self.latest_image_msg: Image | None = None
        self.frame_count: int = 0
        self.keyframe_count: int = 0

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

    def detect_with_superpoint(self, msg: Image) -> tuple[np.ndarray, np.ndarray]:
        gpu_img = self.ros_image_to_tensor(msg)
        feats = self.extractor.extract(gpu_img)
        kps = feats["keypoints"][0].cpu().numpy()  # (n, 2)
        desc = feats["descriptors"][0].cpu().numpy()  # (n, 256)

        img_np = self.ros_image_to_numpy(msg)
        # sky_mask = self.get_sky_mask(img_np) # this feels weird
        kps_int = kps.reshape(-1, 2).astype(int)  # do i need to reshape?

        # if this ever trips, that is a bug in super flow and not us. if this is still
        # here after a while, let's remove it
        # times thought about this: 1
        assert (kps_int[:, 0] >= 0).all() and (kps_int[:, 0] < img_np.shape[1]).all()
        assert (kps_int[:, 1] >= 0).all() and (kps_int[:, 1] < img_np.shape[0]).all()

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

    def initialize_track_ids(self, new_pts: np.ndarray, new_descs: np.ndarray):
        new_ids = list(range(self.track_id, self.track_id + len(new_pts)))
        self.track_id += len(new_pts)
        for tid, desc in zip(new_ids, new_descs):
            self.track_lengths[tid] = 1
            self.track_descriptors[tid] = desc  # [1, 256]
        self.track_ids = new_ids
        self.get_logger().info(f"redetect: first frame, created {len(new_ids)} tracks")
        return new_pts

    def redetect_and_merge(self, msg: Image) -> np.ndarray:
        new_pts, new_descs = self.detect_with_superpoint(msg)

        if self.prev_pts is None:
            return self.initialize_track_ids(new_pts=new_pts, new_descs=new_descs)

        existing = self.prev_pts.reshape(-1, 2)
        candidates = new_pts.reshape(-1, 2)
        merged_pts = list(existing)
        merged_ids = list(self.track_ids)

        new_count = 0
        reassociated_count = 0
        skipped_count = 0

        for pt, desc in zip(candidates, new_descs):
            dists = np.linalg.norm(existing - pt, axis=1)
            if dists.min() > 10:
                tid = self.match_lost_track(desc)
                if tid is None:
                    tid = self.track_id
                    self.track_id += 1
                    new_count += 1
                else:
                    reassociated_count += 1
                self.track_lengths[tid] = self.track_lengths.get(tid, 0) + 1
                self.track_descriptors[tid] = desc
                merged_pts.append(pt)
                merged_ids.append(tid)
            else:
                skipped_count += 1

        # age out lost tracks
        to_remove = []
        for tid in self.lost_track_age:
            self.lost_track_age[tid] += 1
            if self.lost_track_age[tid] > self.max_lost_memory:
                to_remove.append(tid)
        for tid in to_remove:
            # self.get_logger().info(
            #     f"redetect: culling lost track {tid} after {self.max_lost_memory} frames"
            # )
            del self.lost_track_descriptors[tid]
            del self.lost_track_age[tid]

        self.track_ids = merged_ids
        # self.get_logger().info(f"redetect: total merged tracks: {len(merged_ids)}")
        return np.array(merged_pts).reshape(-1, 1, 2).astype(np.float32)

    def yaw_from_pose(self, pose: Pose) -> float:
        # convert quaternion to yaw
        q = pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def position_from_pose(self, pose: Pose) -> np.ndarray:
        return np.array([pose.position.x, pose.position.y, pose.position.z])

    def rpy_from_pose(self, pose: Pose) -> tuple[float, float, float]:
        q = pose.orientation
        # roll
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        # pitch
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
        # yaw
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def match_lost_track(self, desc: np.ndarray, threshold: float = 0.7) -> int | None:
        if len(self.lost_track_descriptors) == 0:
            return None
        lost_ids = list(self.lost_track_descriptors.keys())
        lost_descs = np.stack([self.lost_track_descriptors[tid] for tid in lost_ids])
        desc_t = torch.from_numpy(desc).cuda().unsqueeze(0)
        lost_t = torch.from_numpy(lost_descs).cuda()
        desc_t = torch.nn.functional.normalize(desc_t, dim=1)
        lost_t = torch.nn.functional.normalize(lost_t, dim=1)
        sim = torch.mm(desc_t, lost_t.T).squeeze(0)  # (M,)
        best_idx = sim.argmax().item()
        best_score = sim[best_idx].item()  # ty: ignore

        if best_score < threshold:
            return None

        # mutual nearest neighbor check
        sim_reverse = torch.mm(lost_t[best_idx].unsqueeze(0), desc_t.T).squeeze(0)  # ty: ignore
        if sim_reverse.argmax().item() != 0:  # 0 because desc_t has only one row
            return None  # not a mutual match

        tid = lost_ids[best_idx]  # ty: ignore
        del self.lost_track_descriptors[tid]
        del self.lost_track_age[tid]
        return tid

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

    def store_keyframe(
        self,
        keyframe_id: int,
        pose: PoseStamped,
        descriptors: np.ndarray,
        pts: np.ndarray,
    ):
        if self.ref_sin_lat is None:
            return
        assert self.latest_image_msg is not None
        assert self.latest_odom_msg is not None
        img = self.ros_image_to_numpy(self.latest_image_msg).copy()
        # img_small = cv2.resize(img, (320, 240))
        self.keyframe_db.append(
            {
                "keyframe_id": keyframe_id,
                "pose": pose,
                "descriptors": descriptors,  # (K, 256) for current matched points
                "pts": pts,  # (K, 2) pixel coordinates
                "image": img,  # should this be compressed?
            }
        )
        self.get_logger().info(
            f"storing keyframe! total keyframes: {len(self.keyframe_db)}"
        )

    def find_loop_candidates(
        self,
        current_gps: np.ndarray,
        radius: float = 2.0,
        max_roll_diff: float = np.radians(20),
        max_pitch_diff: float = np.radians(20),
        max_yaw_diff: float = np.radians(20),
    ) -> list[dict]:
        if len(self.keyframe_db) < self.min_keyframes:
            self.get_logger().info(
                f"loop closure: not enough keyframes yet ({len(self.keyframe_db)}/{self.min_keyframes})"
            )
            return []
        assert self.latest_pose is not None
        assert self.latest_odom_msg is not None
        q = self.latest_odom_msg.q
        curr_roll, curr_pitch, curr_yaw = Rotation(quat=[q[1], q[2], q[3], q[0]])
        curr_ned = self.position_from_pose(self.latest_pose.pose)
        candidates = []
        gps_candidates = 0
        separation_candidates = 0
        yaw_candidates = 0
        for kf in self.keyframe_db:
            dist_to_candidate = np.linalg.norm(kf["ned_pos"] - current_gps)
            if dist_to_candidate < radius:
                gps_candidates += 1
                idx = self.keyframe_db.index(kf)
                frames_since = len(self.keyframe_db) - idx
                if frames_since > 10:
                    separation_candidates += 1
                    past_roll, past_pitch, past_yaw = self.rpy_from_pose(
                        kf["pose"]
                    )  # fix keyframe definition first
                    yaw_diff = abs(
                        np.arctan2(
                            np.sin(curr_yaw - past_yaw), np.cos(curr_yaw - past_yaw)
                        )
                    )
                    pitch_diff = abs(
                        np.arctan2(
                            np.sin(curr_pitch - past_pitch),
                            np.cos(curr_pitch - past_pitch),
                        )
                    )
                    roll_diff = abs(
                        np.arctan2(
                            np.sin(curr_roll - past_roll), np.cos(curr_roll - past_roll)
                        )
                    )
                    alt_diff = abs(curr_ned[2] - kf["ned_pos"][2])
                    reason = None
                    if yaw_diff >= max_yaw_diff:
                        reason = f"yaw={np.degrees(yaw_diff):.1f}deg"
                    elif pitch_diff >= max_pitch_diff:
                        reason = f"pitch={np.degrees(pitch_diff):.1f}deg"
                    elif roll_diff >= max_roll_diff:
                        reason = f"roll={np.degrees(roll_diff):.1f}deg"
                    elif alt_diff >= 3.0:
                        reason = f"alt={alt_diff:.1f}m"
                    if reason is None:
                        yaw_candidates += 1
                        candidates.append(kf)
                    else:
                        self.get_logger().info(
                            f"rejected kf {kf['keyframe_id']}: {reason}"
                        )
                else:
                    self.get_logger().info(
                        f"rejected kf {kf['keyframe_id']}: frames_since={frames_since}, "
                        f"(need {self.min_separation}m)"
                    )
        self.get_logger().info(
            f"loop closure search: {len(self.keyframe_db)} keyframes, "
            f"{gps_candidates} within {radius}m, "
            f"{separation_candidates} pass separation, "
            f"{yaw_candidates} pass yaw -> {len(candidates)} candidates"
        )
        return candidates

    def verify_loop_closure(
        self,
        desc_curr: np.ndarray,
        desc_past: np.ndarray,
        pts_curr: np.ndarray,
        pts_past: np.ndarray,
    ) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
        desc_curr_t = torch.nn.functional.normalize(
            torch.from_numpy(desc_curr).cuda(), dim=1
        )
        desc_past_t = torch.nn.functional.normalize(
            torch.from_numpy(desc_past).cuda(), dim=1
        )
        sim = torch.mm(desc_curr_t, desc_past_t.T)  # (K_curr, K_past)

        # forward: best match in past for each current descriptor
        best_matches_fwd = sim.argmax(dim=1)  # (K_curr,)
        best_scores_fwd = sim.max(dim=1).values

        # backward: best match in current for each past descriptor
        best_matches_bwd = sim.argmax(dim=0)  # (K_past,)

        # mutual nearest neighbor: forward match must agree with backward match
        mutual_mask = torch.zeros(len(desc_curr), dtype=torch.bool, device="cuda")
        for i, j in enumerate(best_matches_fwd):
            if best_matches_bwd[j] == i:
                mutual_mask[i] = True

        # combine with score threshold
        good_mask = mutual_mask & (best_scores_fwd > 0.95)

        n_good = good_mask.sum().item()
        self.get_logger().info(
            f"verify loop closure: {n_good} mutual matches out of {len(desc_curr)} "
            f"(max score={best_scores_fwd.max().item():.3f}, "
            f"mean score={best_scores_fwd[good_mask].mean().item() if n_good > 0 else 0:.3f})"
        )

        if n_good < 10:
            return False, None, None

        matched_pts_curr = pts_curr[good_mask.cpu().numpy()]
        matched_pts_past = pts_past[best_matches_fwd[good_mask].cpu().numpy()]
        self.get_logger().info(f"loop closure VERIFIED: {n_good} mutual inliers")
        return True, matched_pts_curr, matched_pts_past

    def should_store_keyframe(self, current_gps: np.ndarray) -> bool:
        if len(self.keyframe_db) == 0:
            return True
        last_gps = self.keyframe_db[-1]["ned_pos"]
        return bool(np.linalg.norm(current_gps - last_gps) > self.min_keyframe_distance)

    def should_publish_keyframe(self) -> bool:
        if self.last_keyframe_pose is None:
            self.last_keyframe_pose = self.update_last_keyframe_pose()
            return False
        if self.latest_pose is None:
            return False
        pos = self.position_from_pose(self.latest_pose.pose)
        translation = np.linalg.norm(pos - self.last_keyframe_pose[:3])
        if translation >= self.keyframe_translation_threshold:
            return True
        # rotation check using quaternion dot product
        q = self.latest_pose.pose.orientation
        q_last = self.last_keyframe_pose[3:]
        dot = abs(q.w * q_last[3] + q.x * q_last[0] + q.y * q_last[1] + q.z * q_last[2])
        dot = np.clip(dot, 0.0, 1.0)
        angle = 2.0 * np.arccos(dot)
        return angle >= self.keyframe_rotation_threshold

    def update_last_keyframe_pose(self) -> np.ndarray | None:
        if self.latest_pose is None:
            return
        q = self.latest_pose.pose.orientation
        pos = self.position_from_pose(self.latest_pose.pose)
        # store as [x, y, z, qx, qy, qz, qw]
        return np.array([pos[0], pos[1], pos[2], q.x, q.y, q.z, q.w])

    # idk about this
    # def get_sky_mask(self, img: np.ndarray) -> np.ndarray:
    #     # img is RGB
    #     hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #     # sky: hue around blue (100-130), low-ish saturation, high value
    #     sky_pixels = (
    #         (hsv[:, :, 0] >= 90)
    #         & (hsv[:, :, 0] <= 140)
    #         & (hsv[:, :, 1] < 100)
    #         & (hsv[:, :, 2] > 150)
    #     )
    #     # also catch white/grey overcast sky
    #     overcast = (hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 200)
    #     sky_pixels = sky_pixels | overcast
    #
    #     # connected components — only keep large regions
    #     sky_uint8 = sky_pixels.astype(np.uint8) * 255
    #     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sky_uint8)
    #     min_area = img.shape[0] * img.shape[1] * 0.01  # at least 1% of image
    #     sky_mask = np.zeros_like(sky_pixels)
    #     for label in range(1, num_labels):  # skip background label 0
    #         if stats[label, cv2.CC_STAT_AREA] >= min_area:
    #             sky_mask |= labels == label
    #
    #     return sky_mask

    def image_callback(self, msg: Image):
        self.get_logger().info("got here")
        gray = self.ros_image_to_gray(msg)

        # redetect with superpoint periodically or on first frame
        if self.prev_gray is None or self.frame_count % self.redetect_every == 0:
            self.prev_pts = self.redetect_and_merge(msg)
            self.prev_gray = gray
            self.latest_image_msg = msg
            self.frame_count += 1
            return

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
            self.prev_gray = None  # force redetection next frame
            self.frame_count += 1
            return

        # ravel is like flatten, but tries to return a view and not a copy
        good_mask = status.ravel() == 1  # == 1 to create boolean mask
        pts1 = curr_pts[good_mask].reshape(-1, 2)
        # note to self: consider numpy array in the future
        track_ids = [tid for tid, ok in zip(self.track_ids, good_mask) if ok]

        # move dead tracks to lost
        dead_ids = set(self.track_ids) - set(track_ids)
        for tid in dead_ids:
            if tid in self.track_descriptors:
                self.lost_track_descriptors[tid] = self.track_descriptors[tid]
                self.lost_track_age[tid] = 0
                del self.track_descriptors[tid]

        # age out lost tracks
        to_remove = [
            tid
            for tid, age in self.lost_track_age.items()
            if age > self.max_lost_memory
        ]
        for tid in to_remove:
            del self.lost_track_descriptors[tid]
            del self.lost_track_age[tid]
        for tid in self.lost_track_age:
            if tid not in to_remove:
                self.lost_track_age[tid] += 1

        # update track lengths and descriptors for survivors
        for tid in track_ids:
            self.track_lengths[tid] = self.track_lengths.get(tid, 0) + 1
        self.track_lengths = {tid: self.track_lengths[tid] for tid in track_ids}
        self.track_descriptors = {
            tid: self.track_descriptors[tid]
            for tid in track_ids
            if tid in self.track_descriptors
        }

        # filter mature tracks
        mature_mask = np.array(
            [
                self.track_lengths.get(tid, 0) >= self.min_track_length
                for tid in track_ids
            ]
        )

        mature_pts1 = None
        mature_descs = None

        if len(track_ids) > 0 and any(mature_mask):
            mature_pts1 = pts1[mature_mask]
            mature_ids = [tid for tid, m in zip(track_ids, mature_mask) if m]
            mature_descs = (
                np.stack(
                    [
                        self.track_descriptors[tid]
                        for tid in mature_ids
                        if tid in self.track_descriptors
                    ]
                )
                if any(tid in self.track_descriptors for tid in mature_ids)
                else None
            )

            if self.should_publish_keyframe():
                msg_out = MatchedPoints()
                msg_out.header.stamp = self.get_clock().now().to_msg()
                msg_out.keyframe_id = self.keyframe_count
                msg_out.points_x = mature_pts1[:, 0].tolist()
                msg_out.points_y = mature_pts1[:, 1].tolist()
                if mature_descs is not None:
                    msg_out.descriptors = mature_descs.flatten().tolist()
                msg_out.track_ids = mature_ids
                self._matched_points_pub.publish(msg_out)
                self.last_keyframe_pose = self.update_last_keyframe_pose()

        # keyframe storage and loop closure
        if (
            self.latest_gps_msg is not None
            and self.latest_pose is not None
            and mature_pts1 is not None
            and mature_descs is not None
        ):
            if self.ref_sin_lat is None:
                self.init_reference(
                    self.latest_gps_msg.latitude_deg,
                    self.latest_gps_msg.longitude_deg,
                    self.latest_gps_msg.altitude_msl_m,
                )
            north, east = self.project(
                self.latest_gps_msg.latitude_deg,
                self.latest_gps_msg.longitude_deg,
            )
            down = -(self.latest_gps_msg.altitude_msl_m - self.ref_alt)
            current_gps = np.array([north, east, down])

            store_keyframe = self.should_store_keyframe(current_gps)
            if store_keyframe:
                self.store_keyframe(
                    keyframe_id=self.keyframe_count,
                    pose=self.latest_pose.pose,
                    descriptors=mature_descs,
                    pts=mature_pts1,
                )
                candidates = self.find_loop_candidates(current_gps)
                for candidate in candidates:
                    success, pts_curr, pts_past = self.verify_loop_closure(
                        desc_curr=mature_descs,
                        desc_past=candidate["descriptors"],
                        pts_curr=mature_pts1,
                        pts_past=candidate["pts"],
                    )
                    if success:
                        lc_msg = LoopClosure()
                        lc_msg.header.stamp = self.get_clock().now().to_msg()
                        lc_msg.current_keyframe_id = self.keyframe_count
                        lc_msg.loop_keyframe_id = candidate["keyframe_id"]
                        lc_msg.points0_x = pts_curr[:, 0].tolist()  # ty: ignore
                        lc_msg.points0_y = pts_curr[:, 1].tolist()  # ty: ignore
                        lc_msg.points1_x = pts_past[:, 0].tolist()  # ty: ignore
                        lc_msg.points1_y = pts_past[:, 1].tolist()  # ty: ignore
                        self._loop_closure_pub.publish(lc_msg)
                        self.get_logger().info(
                            f"loop closure found: {lc_msg.current_keyframe_id} -> {lc_msg.loop_keyframe_id}"
                        )
                        self.get_logger().info(
                            f"keyframes stored: {len(self.keyframe_db)}"
                        )
                        self.get_logger().info(
                            f"lost tracks: {len(self.lost_track_descriptors)}, reassociated: ..."
                        )
                        # get current and past pose positions
                        curr_pos = [
                            self.latest_pose.pose.position.x,
                            self.latest_pose.pose.position.y,
                            self.latest_pose.pose.position.z,
                        ]
                        past_pos = [
                            candidate["pose"].position.x,
                            candidate["pose"].position.y,
                            candidate["pose"].position.z,
                        ]
                        self.get_logger().info(
                            f"pts_curr: {len(pts_curr)}, pts_past: {len(pts_past)}"  # ty: ignore
                        )
                        rr.set_time("keyframe", sequence=self.keyframe_count)
                        rr.log(
                            f"world/loop_closures/{self.keyframe_count}",
                            rr.LineStrips3D(
                                [[curr_pos, past_pos]], colors=[[255, 100, 0]]
                            ),
                        )
                        assert self.latest_image_msg is not None
                        img_curr_np = self.ros_image_to_numpy(
                            self.latest_image_msg
                        ).copy()
                        img_past_np = candidate["image"].copy()

                        for pt in pts_curr:  # ty: ignore
                            cv2.circle(
                                img_curr_np,
                                (int(pt[0]), int(pt[1])),
                                4,
                                (0, 255, 0),
                                -1,
                            )

                        for pt in pts_past:  # ty: ignore
                            cv2.circle(
                                img_past_np,
                                (int(pt[0]), int(pt[1])),
                                4,
                                (0, 255, 0),
                                -1,
                            )

                        rr.set_time("keyframe", sequence=self.keyframe_count)
                        rr.log("loop_closure/current_image", rr.Image(img_curr_np))
                        rr.log("loop_closure/past_image", rr.Image(img_past_np))
                        break

        # update track history for visualization
        for pt, tid in zip(pts1, track_ids):
            x, y = int(pt[0]), int(pt[1])
            if tid not in self.track_history:
                self.track_history[tid] = []
            self.track_history[tid].append((x, y))
            if len(self.track_history[tid]) > self.max_history_len:
                self.track_history[tid].pop(0)

        self.track_history = {
            tid: hist for tid, hist in self.track_history.items() if tid in track_ids
        }

        # draw tracks
        img_curr = self.ros_image_to_numpy(msg).copy()
        # for tid, hist in self.track_history.items():
        #     is_mature = self.track_lengths.get(tid, 0) >= self.min_track_length
        #     color = (0, 255, 255) if is_mature else (0, 255, 0)  # yellow vs green
        #     for i in range(1, len(hist)):
        #         cv2.line(img_curr, hist[i - 1], hist[i], color, 1)
        #     cv2.circle(img_curr, hist[-1], 3, (0, 0, 255), -1)
        #     cv2.putText(
        #         img_curr,
        #         str(tid),
        #         hist[-1],
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.3,
        #         (255, 255, 0),
        #         1,
        #     )
        debug_tid = 0  # match whatever track id backend is tracking
        if debug_tid in self.track_history:
            hist = self.track_history[debug_tid]
            # draw a big circle on the current position
            cv2.circle(img_curr, hist[-1], 10, (255, 0, 255), 2)  # magenta ring
            # draw the full history in magenta
            for i in range(1, len(hist)):
                cv2.line(img_curr, hist[i - 1], hist[i], (255, 0, 255), 2)
            cv2.putText(
                img_curr,
                f"DEBUG tid={debug_tid}",
                (hist[-1][0] + 12, hist[-1][1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                1,
            )

        rr.set_time("keyframe", sequence=self.frame_count)
        rr.log("camera/tracks", rr.Image(img_curr), static=True)

        # update state
        self.prev_gray = gray
        self.prev_pts = curr_pts[good_mask].reshape(-1, 1, 2)
        self.track_ids = track_ids
        self.latest_image_msg = msg
        self.frame_count += 1
        self.keyframe_count += 1

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
