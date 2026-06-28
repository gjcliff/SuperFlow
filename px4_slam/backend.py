import time
from typing import cast

import gtsam
import numpy as np
import rclpy
import rerun as rr
from geometry_msgs.msg import PoseStamped
from gtsam.gtsam.symbol_shorthand import V, X
from px4_msgs.msg import VehicleOdometry
from px4_slam_interfaces.msg import Keyframe as KeyframeMsg
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo

from px4_slam.data import Keyframe


class BackendState:
    def __init__(self):
        self.prev_kf: Keyframe | None = None
        self.smart_factors: dict[int, gtsam.SmartProjectionPoseFactorCal3_S2] = {}

    def reset(self):
        self.prev_kf = None
        self.smart_factors.clear()

    def update(self, kf: Keyframe):
        self.prev_kf = kf

    # @property
    # def needs_redetect(self) -> bool:
    #     return self.prev_pts is None or self.prev_pts.shape[0] < 100


class Backend(Node):
    def __init__(self):
        super().__init__("backend")

        self.declare_parameter("recording_id", str(int(time.time())))
        recording_id = (
            self.get_parameter("recording_id").get_parameter_value().string_value
        )

        rr.init("superflow", recording_id=recording_id)
        rr.connect_grpc()
        rr.log("world", rr.ViewCoordinates.FRD, static=True)

        self._local_position_sub = self.create_subscription(
            VehicleOdometry,
            "fmu/out/vehicle_odometry",
            self.odometry_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self._keyframe_sub = self.create_subscription(
            KeyframeMsg,
            "superflow/keyframe",
            self.keyframe_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self._camera_info_sub = self.create_subscription(
            CameraInfo,
            "camera/camera_info",
            self.camera_info_callback,
            qos_profile=qos_profile_sensor_data,
        )

        self._pose_pub = self.create_publisher(
            PoseStamped, "state_estimate/pose", qos_profile_sensor_data
        )

        self.count: int = 0
        self.prev_imu_timestamp: int | None = None

        self.state = BackendState()
        self.prev_kf: Keyframe | None = None

        self.latest_odom_msg: VehicleOdometry | None = None
        self.isam: gtsam.ISAM2
        self.K: gtsam.Cal3_S2 | None = None
        self.pixel_noise: gtsam.noiseModel.Isotropic
        self.smart_params: gtsam.SmartProjectionParams

        self.trajectory: list[list[float]] = []

        isam_params = gtsam.ISAM2Params()
        isam_params.setRelinearizeThreshold(0.01)
        isam_params.relinearizeSkip = 1
        isam_params.cacheLinearizedFactors = False
        self.isam = gtsam.ISAM2(isam_params)

        smart_params = gtsam.SmartProjectionParams()
        smart_params.setDegeneracyMode(gtsam.DegeneracyMode.ZERO_ON_DEGENERACY)
        smart_params.setRankTolerance(1.0)
        self.smart_params = smart_params

        self.pixel_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.5)
        self._pending_observations: dict[int, tuple[int, np.ndarray]] = {}
        self.smart_factors: dict[int, gtsam.SmartProjectionPoseFactorCal3_S2] = {}
        self.track_pose_keys: dict[int, set[int]] = {}

        body_R_cam = gtsam.Rot3(
            np.array(
                [
                    [0, 0, 1],
                    [1, 0, 0],
                    [0, 1, 0],
                ]
            )
        )
        body_t_cam = gtsam.Point3(0.12, 0.03, 0.242)
        self.body_P_cam: gtsam.Pose3 = gtsam.Pose3(body_R_cam, body_t_cam)

        init_graph = gtsam.NonlinearFactorGraph()
        init_values = gtsam.Values()
        init_graph, init_values = self.set_priors(graph=init_graph, values=init_values)
        self.isam.update(init_graph, init_values)

    def set_priors(
        self, graph: gtsam.NonlinearFactorGraph, values: gtsam.Values
    ) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values]:
        prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        initial_pose = gtsam.Pose3(np.eye(4))
        graph.push_back(gtsam.PriorFactorPose3(X(0), initial_pose, prior_noise))
        values.insert(X(0), initial_pose)

        vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        initial_vel = gtsam.Point3(0.0, 0.0, 0.0)
        graph.push_back(gtsam.PriorFactorVector(V(0), initial_vel, vel_noise))
        values.insert(V(0), initial_vel)
        return graph, values

    def log_pose(self, pose: gtsam.Pose3):
        t = pose.translation()
        q = pose.rotation().toQuaternion()
        rr.set_time("keyframe", sequence=self.count)
        rr.log(
            "world/drone",
            rr.Transform3D(
                translation=[t[0], t[1], t[2]],
                rotation=rr.Quaternion(xyzw=[q.x(), q.y(), q.z(), q.w()]),
            ),
        )
        rr.log("world/drone/axes", rr.TransformAxes3D(axis_length=1.0), static=True)
        self.trajectory.append([t[0], t[1], t[2]])
        rr.log(
            "world/trajectory",
            rr.LineStrips3D([self.trajectory], colors=[[0, 200, 255]]),
        )

    def keyframe_callback(self, msg: KeyframeMsg):
        if self.K is None:
            return
        kf = Keyframe.from_ros_msg(msg)

        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        pose_key = X(kf.kf_id)
        # kf_key = K(kf.kf_id)

        world_T_body = gtsam.Pose3(
            gtsam.Rot3.Quaternion(*kf.q),
            gtsam.Point3(*kf.position),
        )
        values.insert(pose_key, world_T_body)

        if self.state.prev_kf is None:
            self.state.update(kf)
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])
            )
            graph.add(gtsam.PriorFactorPose3(pose_key, world_T_body, prior_noise))
            self.isam.update(graph, values)
            return


        prev_pose_key = X(self.state.prev_kf.kf_id)
        prev_T_world = gtsam.Pose3(
            gtsam.Rot3.Quaternion(*self.state.prev_kf.q),
            gtsam.Point3(*self.state.prev_kf.position),
        )

        relative_pose = prev_T_world.inverse().compose(world_T_body)
        odom_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3])
        )
        graph.add(
            gtsam.BetweenFactorPose3(
                prev_pose_key,
                pose_key,
                relative_pose,
                odom_noise,
            )
        )

        for i, track_id in enumerate(kf.track_ids):
            pt = kf.kps[i]
            measurement = gtsam.Point2(pt[0], pt[1])

            if track_id not in self.state.smart_factors:
                self.state.smart_factors[track_id] = (
                    gtsam.SmartProjectionPoseFactorCal3_S2(
                        self.pixel_noise, self.K, self.body_P_cam, self.smart_params
                    )
                )
                graph.push_back(self.state.smart_factors[track_id])

            try:
                self.state.smart_factors[track_id].add(measurement, pose_key)
            except ValueError:
                breakpoint()

        self.isam.update(graph, values)
        self.state.update(kf)

        result = cast(gtsam.Values, self.isam.calculateEstimate())
        optimized_pose = result.atPose3(pose_key)
        self.log_pose(optimized_pose)

        points = []
        for factor in self.state.smart_factors.values():
            point = factor.point(result)
            if point is not None:
                points.append(np.array(point))
        if points:
            rr.log(
                "world/landmarks",
                rr.Points3D(
                    np.array(points),
                    colors=[[0, 255, 255]] * len(points),
                    radii=0.05,
                ),
            )

    def log_kf_pinhole(self, kf: Keyframe):
        if self.K is None:
            return
        q = kf.q  # [w, x, y, z]
        world_R_body = gtsam.Rot3.Quaternion(q[0], q[1], q[2], q[3])
        world_t_body = gtsam.Point3(*kf.position)
        world_T_body = gtsam.Pose3(world_R_body, world_t_body)

        # camera pose in world frame
        world_T_cam = world_T_body.compose(self.body_P_cam)

        # extract for rerun
        t = world_T_cam.translation()
        q_cam = (
            world_T_cam.rotation().toQuaternion()
        )  # gtsam quaternion is [w, x, y, z]

        rr.log(
            f"world/camera/keyframes/{kf.kf_id}",
            rr.Transform3D(
                translation=np.array([t[0], t[1], t[2]]),
                rotation=rr.Quaternion(
                    xyzw=[q_cam.x(), q_cam.y(), q_cam.z(), q_cam.w()]
                ),
            ),
        )
        rr.log(
            f"world/camera/keyframes/{kf.kf_id}/pinhole",
            rr.Pinhole(
                focal_length=(self.K.fx(), self.K.fy()),
                principal_point=(self.K.px(), self.K.py()),
                width=kf.img_size[1],
                height=kf.img_size[0],
            ),
        )

    # ------------------------------------------------------------------
    # sensor callbacks
    # ------------------------------------------------------------------
    def odometry_callback(self, msg: VehicleOdometry):
        self.latest_odom_msg = msg
        rr.log(
            "world/camera/pose",
            rr.Transform3D(
                translation=msg.position,
                rotation=rr.Quaternion(xyzw=[msg.q[1], msg.q[2], msg.q[3], msg.q[0]]),
            ),
            static=True,
        )
        rr.log(
            "world/camera/pose/axes",
            rr.TransformAxes3D(axis_length=1.0),
            static=True,
        )

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
    backend = Backend()
    rclpy.spin(backend)
    backend.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
