import time

import gtsam
import numpy as np
import rclpy
import rerun as rr
from geometry_msgs.msg import PoseStamped
from gtsam.symbol_shorthand import B, V, X
from px4_msgs.msg import SensorCombined, SensorGps, VehicleMagnetometer
from px4_slam_interfaces.msg import LoopClosure, MatchedPoints
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo


class Backend(Node):

    initialized: bool = False

    # imu-driven counter — incremented every ~10ms in imu_callback
    count: int = 0
    prev_imu_timestamp: int | None = None

    def __init__(self):
        super().__init__("backend")

        self.declare_parameter("recording_id", str(int(time.time())))
        recording_id = (
            self.get_parameter("recording_id").get_parameter_value().string_value
        )

        rr.init("super_flow", recording_id=recording_id)
        rr.connect_grpc()
        rr.log("world", rr.ViewCoordinates.FRD, static=True)

        self._imu_sub = self.create_subscription(
            SensorCombined,
            "fmu/out/sensor_combined",
            self.imu_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self._gps_sub = self.create_subscription(
            SensorGps,
            "fmu/out/sensor_gps",
            self.gps_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self._magnetometer_sub = self.create_subscription(
            VehicleMagnetometer,
            "fmu/out/vehicle_magnetometer",
            self.magnetometer_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self._matched_points_sub = self.create_subscription(
            MatchedPoints,
            "camera/matched_points",
            self.matched_points_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self._loop_closure_sub = self.create_subscription(
            LoopClosure,
            "camera/loop_closure",
            self.loop_closure_callback,
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

        self.pim: gtsam.PreintegratedImuMeasurements
        self.biasKey: int
        self.biasNoise: gtsam.noiseModel.Isotropic
        self.latest_gps_msg: SensorGps | None = None
        self.latest_mag_msg: VehicleMagnetometer | None = None
        self.ref_sin_lat: float | None = None
        self.ref_cos_lat: float | None = None
        self.ref_lat: float | None = None
        self.ref_lon: float | None = None
        self.ref_alt: float | None = None
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
        init_graph, init_values = self.setup_imu(graph=init_graph, values=init_values)
        self.isam.update(init_graph, init_values)
        self.initialized = True

    # ------------------------------------------------------------------
    # graph initialisation helpers
    # ------------------------------------------------------------------

    def set_priors(
        self, graph: gtsam.NonlinearFactorGraph, values: gtsam.Values
    ) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values]:
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        )
        initial_pose = gtsam.Pose3(np.eye(4))
        graph.push_back(gtsam.PriorFactorPose3(X(0), initial_pose, prior_noise))
        values.insert(X(0), initial_pose)

        vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        initial_vel = gtsam.Point3(0.0, 0.0, 0.0)
        graph.push_back(gtsam.PriorFactorVector(V(0), initial_vel, vel_noise))
        values.insert(V(0), initial_vel)
        return graph, values

    def setup_imu(self, graph: gtsam.NonlinearFactorGraph, values: gtsam.Values):
        self.biasKey = B(0)
        self.biasNoise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
        graph.push_back(
            gtsam.PriorFactorConstantBias(
                self.biasKey, gtsam.imuBias.ConstantBias(), self.biasNoise
            )
        )
        values.insert(self.biasKey, gtsam.imuBias.ConstantBias())
        pim_params = self.pim_params()
        self.pim = gtsam.PreintegratedImuMeasurements(pim_params)
        return graph, values

    def pim_params(self):
        params = gtsam.PreintegrationParams.MakeSharedD(9.81)
        I = np.eye(3)  # noqa: E741
        params.setAccelerometerCovariance(I * 0.01)
        params.setGyroscopeCovariance(I * 0.001)
        params.setIntegrationCovariance(I * 0.01)
        params.setUse2ndOrderCoriolis(False)
        params.setOmegaCoriolis(np.array([0, 0, 0], dtype=float))
        return params

    # ------------------------------------------------------------------
    # gps / mag helpers
    # ------------------------------------------------------------------

    def init_reference(self, lat_0, lon_0, alt_0):
        self.ref_alt = alt_0
        self.ref_lat = np.radians(lat_0)
        self.ref_lon = np.radians(lon_0)
        self.ref_sin_lat = np.sin(self.ref_lat)
        self.ref_cos_lat = np.cos(self.ref_lat)

    def project_gps(self, lat, lon):
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

    def add_gps_factor(
        self, graph: gtsam.NonlinearFactorGraph, key: int, msg: SensorGps
    ):
        if self.ref_sin_lat is None:
            self.init_reference(msg.latitude_deg, msg.longitude_deg, msg.altitude_msl_m)
        north, east = self.project_gps(msg.latitude_deg, msg.longitude_deg)
        down = -(msg.altitude_msl_m - self.ref_alt)
        gps = gtsam.Point3(north, east, down)
        gps_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        graph.add(gtsam.GPSFactor(key, gps, gps_noise))
        return north, east, down

    def add_mag_factor(
        self, graph: gtsam.NonlinearFactorGraph, key: int, msg: VehicleMagnetometer
    ):
        yaw = np.arctan2(-msg.magnetometer_ga[1], msg.magnetometer_ga[0])
        rot = gtsam.Rot3.Yaw(yaw)
        pose = gtsam.Pose3(rot, gtsam.Point3(0, 0, 0))
        mag_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 1e6, 1e6, 1e6])
        )
        graph.add(gtsam.PriorFactorPose3(key, pose, mag_noise))

    # ------------------------------------------------------------------
    # logging / publishing
    # ------------------------------------------------------------------

    def publish_pose(self, pose: gtsam.Pose3):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        t = pose.translation()
        q = pose.rotation().toQuaternion()
        msg.pose.position.x = t[0]
        msg.pose.position.y = t[1]
        msg.pose.position.z = t[2]
        msg.pose.orientation.x = q.x()
        msg.pose.orientation.y = q.y()
        msg.pose.orientation.z = q.z()
        msg.pose.orientation.w = q.w()
        self._pose_pub.publish(msg)

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

    # ------------------------------------------------------------------
    # imu callback — runs at ~100hz, owns the pose counter and graph
    # ------------------------------------------------------------------

    def imu_callback(self, msg: SensorCombined) -> None:
        if not self.initialized:
            return
        if self.prev_imu_timestamp is None:
            self.prev_imu_timestamp = msg.timestamp
            return

        dt = msg.accelerometer_integral_dt * 1e-6
        accel = np.array(msg.accelerometer_m_s2)
        gyro = np.array(msg.gyro_rad)
        self.pim.integrateMeasurement(accel, gyro, dt)

        # update at ~100hz (50ms gate, same as state_estimation.py)
        if msg.timestamp - self.prev_imu_timestamp < 50_000:
            return

        i = self.count

        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        pose_i = self.isam.calculateEstimatePose3(X(i))
        vel_i = self.isam.calculateEstimateVector(V(i))
        bias_i = self.isam.calculateEstimateConstantBias(self.biasKey)
        prev_state = gtsam.NavState(pose_i, vel_i)
        pred_state = self.pim.predict(prev_state, bias_i)

        values.insert(X(i + 1), pred_state.pose())
        values.insert(V(i + 1), pred_state.velocity())

        graph.add(
            gtsam.ImuFactor(X(i), V(i), X(i + 1), V(i + 1), self.biasKey, self.pim)
        )
        self.pim.resetIntegration()

        if self.latest_gps_msg is not None:
            self.add_gps_factor(graph, X(i + 1), self.latest_gps_msg)
            self.latest_gps_msg = None
        if self.latest_mag_msg is not None:
            self.add_mag_factor(graph, X(i + 1), self.latest_mag_msg)
            self.latest_mag_msg = None

        # bias random walk every 50 imu steps (~5s)
        if self.count % 50 == 0 and self.count != 0:
            graph.add(
                gtsam.BetweenFactorConstantBias(
                    self.biasKey,
                    self.biasKey + 1,
                    gtsam.imuBias.ConstantBias(),
                    self.biasNoise,
                )
            )
            self.biasKey += 1
            values.insert(self.biasKey, gtsam.imuBias.ConstantBias())

        self.isam.update(graph, values)

        pose = self.isam.calculateEstimatePose3(X(i + 1))
        self.log_pose(pose)
        self.publish_pose(pose)

        self.prev_imu_timestamp = msg.timestamp
        self.count += 1

    # ------------------------------------------------------------------
    # matched points callback — attaches visual factors to existing pose keys
    # the imu graph has already created and initialized X(count) before this
    # is ever called, so we never need to insert new variables here
    # ------------------------------------------------------------------

    def matched_points_callback(self, msg: MatchedPoints) -> None:
        if not self.initialized:
            return
        if self.K is None:
            self.get_logger().warn("no camera calibration yet, dropping keyframe")
            return

        # snapshot the current imu pose index — this is the pose key we'll
        # attach observations to. the imu callback may increment count while
        # we're running but that's fine; we just use whatever is current now.
        current_pose_key = self.count

        pts_x: list[float] = msg.points_x
        pts_y: list[float] = msg.points_y
        track_ids: list[int] = list(msg.track_ids)

        new_factor_graph = gtsam.NonlinearFactorGraph()

        for tid, u, v in zip(track_ids, pts_x, pts_y):
            observation = np.array([u, v])

            if tid not in self.smart_factors:
                if tid in self._pending_observations:
                    # second observation — promote to a real smart factor
                    prev_key, prev_obs = self._pending_observations.pop(tid)
                    factor = gtsam.SmartProjectionPoseFactorCal3_S2(
                        self.pixel_noise,
                        self.K,
                        self.body_P_cam,
                        self.smart_params,
                    )
                    factor.add(prev_obs, X(prev_key))
                    factor.add(observation, X(current_pose_key))
                    self.smart_factors[tid] = factor
                    self.track_pose_keys[tid] = {prev_key, current_pose_key}
                    new_factor_graph.push_back(factor)
                else:
                    # first observation — buffer and wait for a second keyframe
                    self._pending_observations[tid] = (current_pose_key, observation)
                continue

            # existing factor — add the new observation if this pose key is new
            factor = self.smart_factors[tid]
            if current_pose_key not in self.track_pose_keys[tid]:
                factor.add(observation, X(current_pose_key))
                self.track_pose_keys[tid].add(current_pose_key)

        if new_factor_graph.size() > 0:
            try:
                self.isam.update(new_factor_graph, gtsam.Values())
            except RuntimeError as e:
                self.get_logger().error(
                    f"isam update (new smart factors) failed at pose {current_pose_key}: {e}"
                )
                self._reset_visual()
                return

        # re-linearize mutated smart factors
        try:
            self.isam.update(gtsam.NonlinearFactorGraph(), gtsam.Values())
        except RuntimeError as e:
            self.get_logger().error(
                f"isam re-linearization failed at pose {current_pose_key}: {e}"
            )
            self._reset_visual()
            return

        # debug logging for track 0
        if self.smart_factors:
            tid = next(iter(self.smart_factors))
            factor = self.smart_factors[tid]
            n_obs = len(self.track_pose_keys[tid])
            current_values = self.isam.calculateEstimate()
            result = factor.point(current_values)
            pose = self.isam.calculateEstimatePose3(X(current_pose_key))
            t = pose.translation()

            if result.valid():
                status_str = "valid"
            elif result.degenerate():
                status_str = "degen"
            elif result.behindCamera():
                status_str = "behind"
            elif result.farPoint():
                status_str = "far"
            elif result.outlier():
                status_str = "outlier"
            else:
                status_str = f"unknown({result.status})"

            self.get_logger().info(
                f"track {tid}: {n_obs} obs, "
                f"pose=({t[0]:.2f},{t[1]:.2f},{t[2]:.2f}), "
                f"status={status_str}"
            )

            if result.valid() or result.farPoint():
                p = result.get()
                cam_pose = pose.compose(self.body_P_cam)
                p_cam = cam_pose.transformTo(gtsam.Point3(float(p[0]), float(p[1]), float(p[2])))
                dist = float(np.linalg.norm([p_cam[0], p_cam[1], p_cam[2]]))
                self.get_logger().info(
                    f"  point=({float(p[0]):.2f},{float(p[1]):.2f},{float(p[2]):.2f}), "
                    f"dist_from_cam={dist:.1f}m"
                )
                if self.K is not None and p_cam[2] > 0.1:
                    u_proj = self.K.fx() * p_cam[0] / p_cam[2] + self.K.px()
                    v_proj = self.K.fy() * p_cam[1] / p_cam[2] + self.K.py()
                    self.get_logger().info(
                        f"  reprojected pixel=({u_proj:.1f}, {v_proj:.1f})"
                    )
                    rr.set_time("keyframe", sequence=current_pose_key)
                    rr.log("camera/debug_projection", rr.Points2D(
                        [[u_proj, v_proj]],
                        radii=8.0,
                        colors=[[255, 80, 0] if status_str != "valid" else [0, 255, 80]],
                        labels=[f"tid={tid} {status_str} d={dist:.0f}m"],
                    ))
                rr.set_time("keyframe", sequence=current_pose_key)
                rr.log("world/points", rr.Points3D(
                    [[float(p[0]), float(p[1]), float(p[2])]],
                    colors=[[255, 80, 0] if status_str != "valid" else [255, 200, 0]],
                ))

    # ------------------------------------------------------------------
    # visual-only reset - drops smart factors without touching the imu graph.
    # the pose variables remain valid so the imu graph keeps running cleanly.
    # ------------------------------------------------------------------

    def _reset_visual(self):
        self.smart_factors = {}
        self.track_pose_keys = {}
        self._pending_observations = {}
        self.get_logger().warn(
            "visual factors reset — imu/gps graph intact, continuing from pose X(%d)",
            self.count,
        )


    # loop closure
    def loop_closure_callback(self, msg: LoopClosure) -> None:
        curr_key = msg.current_keyframe_id
        loop_key = msg.loop_keyframe_id

        if curr_key > self.count or loop_key > self.count:
            self.get_logger().warn(
                f"loop closure references future keyframe ({curr_key}, {loop_key}), dropping"
            )
            return

        pose_curr = self.isam.calculateEstimatePose3(X(curr_key))
        pose_loop = self.isam.calculateEstimatePose3(X(loop_key))
        between = pose_loop.between(pose_curr)

        between_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3])
        )
        factor = gtsam.BetweenFactorPose3(
            X(loop_key), X(curr_key), between, between_noise
        )
        graph = gtsam.NonlinearFactorGraph()
        graph.add(factor)
        self.isam.update(graph, gtsam.Values())
        self.get_logger().info(f"loop closure added: X({loop_key}) -> X({curr_key})")

    # ------------------------------------------------------------------
    # sensor callbacks
    # ------------------------------------------------------------------

    def gps_callback(self, msg: SensorGps):
        self.latest_gps_msg = msg

    def magnetometer_callback(self, msg: VehicleMagnetometer):
        self.latest_mag_msg = msg

    def camera_info_callback(self, msg: CameraInfo):
        if self.K is None:
            self.K = gtsam.Cal3_S2(
                msg.k[0],   # fx
                msg.k[4],   # fy
                msg.k[1],   # s (skew)
                msg.k[2],   # cx
                msg.k[5],   # cy
            )


def main(args=None):
    rclpy.init(args=args)
    backend = Backend()
    rclpy.spin(backend)
    backend.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
