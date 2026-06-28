from dataclasses import asdict, dataclass
from typing import ClassVar, Self

import numpy as np
import rclpy.clock
import rerun as rr
from px4_slam_interfaces.msg import Keyframe as KeyframeMsg
from px4_slam_interfaces.msg import LoopClosure as LoopClosureMsg


@dataclass
class LoopClosure:
    query_kf_id: int
    match_kf_id: int
    rel_pos: np.ndarray
    rel_q: np.ndarray
    n_inliers: int

    def to_ros_msg(self, clock: rclpy.clock.Clock) -> LoopClosureMsg:
        msg = LoopClosureMsg()
        msg.header.stamp = clock.now().to_msg()
        msg.header.frame_id = "world"
        msg.query_kf_id = self.query_kf_id
        msg.match_kf_id = self.match_kf_id
        msg.rel_pos = self.rel_pos
        msg.rel_q = self.rel_q
        msg.n_inliers = self.n_inliers
        return msg

    @classmethod
    def from_ros_msg(cls, msg: LoopClosureMsg) -> Self:
        return cls(
            query_kf_id=msg.query_kf_id,
            match_kf_id=msg.match_kf_id,
            rel_pos=msg.rel_pos,
            rel_q=msg.rel_q,
            n_inliers=msg.n_inliers,
        )


@dataclass
class Keyframe:
    kps: np.ndarray  # [n, 2]
    desc: np.ndarray  # [n, 256]
    position: np.ndarray  # [3,]
    q: np.ndarray  # [1, 4] wxyz
    track_ids: np.ndarray  # [n, 1]
    kf_id: int = 0
    img_size: tuple[int, int] = (0, 0)
    log: bool = False

    _next_id: ClassVar[int] = 0

    def __post_init__(self):
        self.kf_id += Keyframe._next_id
        Keyframe._next_id += 1
        if self.log:
            self.log_keyframe_txt()

    def is_close(
        self,
        position: np.ndarray,
        q: np.ndarray,
        pos_thresh: float = 5.0,
        orientation_thresh: float = 0.9,
    ) -> bool:
        if np.linalg.norm(self.position - position) > pos_thresh:
            return False
        return np.abs(np.dot(self.q, q)) > orientation_thresh

    def get_descs_for_tracks(
        self, track_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = np.isin(track_ids, self.track_ids)
        sorter = np.argsort(self.track_ids)
        idx = sorter[np.searchsorted(self.track_ids, track_ids[mask], sorter=sorter)]
        return mask, self.desc[idx]

    def log_keyframe_txt(self):
        rr.log(
            f"keyframes/{self.kf_id}/kps/length",
            rr.TextLog(self.kps.shape[0]),
            static=True,
        )
        rr.log(
            f"keyframes/{self.kf_id}/desc/length",
            rr.TextLog(self.desc.shape[0]),
            static=True,
        )
        rr.log(
            f"keyframes/{self.kf_id}/pos/x", rr.TextLog(self.position[0]), static=True
        )
        rr.log(
            f"keyframes/{self.kf_id}/pos/y", rr.TextLog(self.position[1]), static=True
        )
        rr.log(
            f"keyframes/{self.kf_id}/pos/z", rr.TextLog(self.position[2]), static=True
        )
        rr.log(f"keyframes/{self.kf_id}/q/w", rr.TextLog(self.q[0]), static=True)
        rr.log(f"keyframes/{self.kf_id}/q/x", rr.TextLog(self.q[0]), static=True)
        rr.log(f"keyframes/{self.kf_id}/q/y", rr.TextLog(self.q[0]), static=True)
        rr.log(f"keyframes/{self.kf_id}/q/z", rr.TextLog(self.q[0]), static=True)
        rr.log(
            f"keyframes/{self.kf_id}/tracks/num_ids",
            rr.TextLog(self.track_ids.shape[0]),
            static=True,
        )
        rr.log(
            f"keyframes/{self.kf_id}/tracks/num_ids",
            rr.TextLog(self.track_ids.shape[0]),
            static=True,
        )

    @classmethod
    def from_ros_msg(cls, msg: KeyframeMsg) -> Self:
        return cls(
            kf_id=msg.kf_id,
            kps=np.column_stack([msg.kps_x, msg.kps_y]),
            desc=np.empty(0),
            position=np.array(msg.position),
            q=np.array(msg.q),
            track_ids=np.array(msg.track_ids),
            img_size=msg.img_size,
            log=False,
        )

    def to_ros_msg(self, clock: rclpy.clock.Clock) -> KeyframeMsg:
        msg = KeyframeMsg()
        msg.header.stamp = clock.now().to_msg()
        msg.header.frame_id = "world"
        msg.kf_id = self.kf_id
        msg.kps_x = self.kps[:, 0].astype(np.float32).tolist()
        msg.kps_y = self.kps[:, 1].astype(np.float32).tolist()
        msg.track_ids = self.track_ids.astype(np.uint32).tolist()
        msg.position = self.position.astype(np.float32).tolist()
        msg.img_size = self.img_size
        msg.q = self.q.astype(np.float32).tolist()
        msg.log = self.log

        return msg

    @classmethod
    def get_next_id(cls):
        return cls._next_id

    def to_dict(self):
        return asdict(self)

    def to_dict_np(self):
        return {
            "kps": self.kps.tolist(),
            "desc": self.desc.tolist(),
            # "position": self.position,
            # "rpy": self.rpy,
        }


class IDGenerator:
    _next_id = 0

    @classmethod
    def next(cls) -> int:
        next_id = cls._next_id
        cls._next_id += 1
        return next_id

    @classmethod
    def next_batch(cls, n: int) -> np.ndarray:
        ids = np.arange(cls._next_id, cls._next_id + n, dtype=np.int32)
        cls._next_id += n
        return ids
