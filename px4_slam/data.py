from dataclasses import asdict, dataclass
from typing import ClassVar

import numpy as np
import rerun as rr


@dataclass
class Keyframe:
    kps: np.ndarray  # [n, 2]
    desc: np.ndarray  # [n, 256]
    position: np.ndarray  # [3,]
    q: np.ndarray  # [1, 4] wxyz
    rot: np.ndarray  # [3, 3]
    track_ids: np.ndarray  # [n, 1]
    kf_id: int = 0
    log: bool = False

    _next_id: ClassVar[int] = 0

    def __post_init__(self):
        self.kf_id += Keyframe._next_id
        Keyframe._next_id += 1
        if self.log:
            self.log_keyframe_txt()

    def log_keyframe_img(self, img: np.ndarray):
        rr.log(f"keyframes/{self.kf_id}/img", rr.Image(img), static=True)

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
