import cv2
import gtsam
import numpy as np
import rerun as rr

from px4_slam.data import Keyframe


def log_loop_closure(new_kf: Keyframe, match_kf: Keyframe, n_inliers: int):
    mid = (new_kf.position + match_kf.position) / 2

    rr.log(
        f"world/loop_closures/{new_kf.kf_id}_{match_kf.kf_id}",
        rr.LineStrips3D(
            [np.array([new_kf.position, match_kf.position])],
            colors=[[0, 255, 0]],  # green for confirmed
            radii=0.02,
        ),
    )
    rr.log(
        f"world/loop_closure_labels/{new_kf.kf_id}",
        rr.Points3D([mid], labels=[f"LC {n_inliers} inliers"], radii=0.1),
    )


def log_lc_candidates(new_kf: Keyframe, candidates: list[Keyframe]):
    lines = []
    for kf in candidates:
        lines.append(np.array([new_kf.position, kf.position]))
    rr.log(
        "world/camera/loop_closure_lines",
        rr.LineStrips3D(lines, colors=[[255, 165, 0]]),
        static=True,
    )


def log_kf(
    K: gtsam.Cal3_S2,
    body_T_cam: np.ndarray,
    kf: Keyframe,
    img: np.ndarray,
    track_counts: np.ndarray,
):
    q = kf.q  # [w, x, y, z]
    world_R_body = gtsam.Rot3.Quaternion(q[0], q[1], q[2], q[3])
    world_t_body = gtsam.Point3(*kf.position)
    world_T_body = gtsam.Pose3(world_R_body, world_t_body)

    world_T_cam = world_T_body.compose(gtsam.Pose3(body_T_cam))

    t = world_T_cam.translation()
    q_cam = world_T_cam.rotation().toQuaternion()  # gtsam quaternion is [w, x, y, z]

    rr.log(
        f"world/camera/keyframes/{kf.kf_id}",
        rr.Transform3D(
            translation=np.array([t[0], t[1], t[2]]),
            rotation=rr.Quaternion(xyzw=[q_cam.x(), q_cam.y(), q_cam.z(), q_cam.w()]),
        ),
    )
    rr.log(
        f"world/camera/keyframes/{kf.kf_id}/pinhole",
        rr.Pinhole(
            focal_length=(K.fx(), K.fy()),
            principal_point=(K.px(), K.py()),
            width=kf.img_size[1],
            height=kf.img_size[0],
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )
    pretty = img.copy()
    for pt, tid in zip(kf.kps, kf.track_ids):
        x, y = (int(pt[0]), int(pt[1]))
        count = track_counts[tid-1]
        cv2.putText(
            pretty,
            f"{tid}, {count}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 255, 0),
            1,
        )
        cv2.circle(pretty, (x, y), radius=3, color=(0, 255, 0), thickness=2)
    rr.log(f"world/camera/keyframes/{kf.kf_id}/pinhole/image", rr.Image(pretty))
