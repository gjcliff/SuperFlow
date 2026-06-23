from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Keyframe:
    kps: np.ndarray  # [n, 2]
    desc: np.ndarray  # [n, 256]
    # position: np.ndarray # [1, 3]
    # rpy: np.ndarray # [1, 3]
    tracks: np.ndarray  # [n, 1]

    def to_dict(self):
        return asdict(self)

    def to_dict_np(self):
        return {
            "kps": self.kps.tolist(),
            "desc": self.desc.tolist(),
            "tracks": self.tracks.tolist(),
            # "position": self.position,
            # "rpy": self.rpy,
        }
