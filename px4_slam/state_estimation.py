import time
import uuid
from collections import defaultdict

import gtsam
import numpy as np
import rclpy
import rerun as rr
from geometry_msgs.msg import PoseStamped
from gtsam.symbol_shorthand import B, V, X
from px4_msgs.msg import (
    SensorCombined,
    SensorGps,
    VehicleAttitude,
    VehicleLocalPosition,
    VehicleMagnetometer,
)
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class Trajectory:
    def __init__(self, rng: np.random.Generator):
        self.points: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.color: list[int] = [
            int(rng.integers(256)),
            int(rng.integers(256)),
            int(rng.integers(256)),
        ]
        self.count: int = 0


class StateEstimation(Node):
    def __init__(self):
        super().__init__("px4_slam")

        self.end_time: int | float = 0
        self.rng = np.random.default_rng(42)
        self.latest_pose: gtsam.Pose3 | None = None

        self.biasKey: int = B(0)
        self.biasNoise = gtsam.noiseModel.Isotropic.Sigma(6, 0.3)
        self.gpsNoise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
        self.magNoise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e6, 1e6, 0.1, 1e6, 1e6, 1e6])
        )
        self.pim: gtsam.PreintegratedImuMeasurements | None = None
        self.latest_gps_msg: SensorGps | None = None
        self.latest_mag_msg: VehicleMagnetometer | None = None
        self.trajectories: dict[str, Trajectory] = defaultdict(
            lambda: Trajectory(self.rng)
        )
        self.imu_meas_count: int = 0
        self.update_every_n: int = 12
        self.key_count: int = 0

        self.ref_sin_lat: float | None = None
        self.ref_cos_lat: float | None = None
        self.ref_lat: float | None = None
        self.ref_lon: float | None = None
        self.ref_alt: float | None = None

        self.declare_parameter("recording_id", str(uuid.uuid4()))
        recording_id = (
            self.get_parameter("recording_id").get_parameter_value().string_value
        )

        rr.init("super_flow", recording_id=recording_id)
        rr.spawn()
        rr.log("world", rr.ViewCoordinates.FRD, static=True)

        self._imu_sub: rclpy.node.Subscription = self.create_subscription(
            SensorCombined,
            "fmu/out/sensor_combined",
            self.imu_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self._gps_sub: rclpy.node.Subscription = self.create_subscription(
            SensorGps,
            "fmu/out/sensor_gps",
            self.gps_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self._magnetometer_sub: rclpy.node.Subscription = self.create_subscription(
            VehicleMagnetometer,
            "fmu/out/vehicle_magnetometer",
            self.magnetometer_callback,
            qos_profile=qos_profile_sensor_data,
        )
        # TODO: implement this and compare my estimate to ground truth
        # maybe this can be a separate node, or a test
        self._local_position_sub: rclpy.node.Subscription = self.create_subscription(
            VehicleLocalPosition,
            "fmu/out/vehicle_local_position",
            self.local_position_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self._attitude_sub: rclpy.node.Subscription = self.create_subscription(
            VehicleAttitude,
            "fmu/out/vehicle_attitude",
            self.attitude_callback,
            qos_profile=qos_profile_sensor_data,
        )

        self._pose_pub = self.create_publisher(
            PoseStamped, "state_estimate/pose", qos_profile_sensor_data
        )

        parameters = gtsam.ISAM2Params()
        # parameters.setRelinearizeThreshold(0.01)
        # parameters.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(parameters)

        init_graph = gtsam.NonlinearFactorGraph()
        init_values = gtsam.Values()
        init_graph, init_values = self.set_priors(graph=init_graph, values=init_values)
        init_graph, init_values = self.setup_imu_preintegration(
            graph=init_graph, values=init_values
        )
        self.isam.update(init_graph, init_values)

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

    def add_gps_factor(
        self, graph: gtsam.NonlinearFactorGraph, key: int, msg: SensorGps
    ):
        if self.ref_sin_lat is None:
            self.init_reference(msg.latitude_deg, msg.longitude_deg, msg.altitude_msl_m)

        north, east = self.project(msg.latitude_deg, msg.longitude_deg)
        down = -(msg.altitude_msl_m - self.ref_alt)
        gps = gtsam.Point3(north, east, down)
        self.log_pose(gtsam.Pose3(gtsam.Rot3(), gps), entity="gps")
        # self.get_logger().info(
        #     f"gps position: north={north:.2f}, east={east:.2f}, down={down:.2f}"
        # )
        graph.add(gtsam.GPSFactor(key, gps, self.gpsNoise))

    def add_mag_factor(
        self,
        graph: gtsam.NonlinearFactorGraph,
        key: int,
        msg: VehicleMagnetometer,
        position: np.ndarray,
    ):
        yaw = np.arctan2(-msg.magnetometer_ga[1], msg.magnetometer_ga[0])
        rot = gtsam.Rot3.Yaw(yaw)
        pose = gtsam.Pose3(rot, position)
        graph.add(gtsam.PriorFactorPose3(key, pose, self.magNoise))

    def set_priors(self, graph: gtsam.NonlinearFactorGraph, values: gtsam.Values):
        priorNoise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 1e6, 1e6, 1e6])
        )
        initial_pose = gtsam.Pose3(
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )
        graph.push_back(gtsam.PriorFactorPose3(X(0), initial_pose, priorNoise))
        values.insert(X(0), initial_pose)

        initial_velocity = gtsam.Point3(np.array([0.0, 0.0, 0.0]))
        velNoise = gtsam.noiseModel.Isotropic.Sigma(3, 0.5)
        graph.push_back(gtsam.PriorFactorVector(V(0), initial_velocity, velNoise))
        values.insert(V(0), initial_velocity)

        return graph, values

    def setup_imu_preintegration(
        self, graph: gtsam.NonlinearFactorGraph, values: gtsam.Values
    ):
        graph.push_back(
            gtsam.PriorFactorConstantBias(
                self.biasKey, gtsam.imuBias.ConstantBias(), self.biasNoise
            )
        )
        values.insert(self.biasKey, gtsam.imuBias.ConstantBias())

        pim_params = self.preintegration_parameters()
        self.pim = gtsam.PreintegratedImuMeasurements(pim_params)

        return graph, values

    def preintegration_parameters(self):
        params = gtsam.PreintegrationParams.MakeSharedD(9.81)
        I = np.eye(3)  # noqa: E741
        params.setAccelerometerCovariance(I * 0.1)
        params.setGyroscopeCovariance(I * 0.1)
        params.setIntegrationCovariance(I * 0.1)
        params.setUse2ndOrderCoriolis(False)
        params.setOmegaCoriolis(np.array([0, 0, 0], dtype=float))
        return params

    def publish_pose(self, pose: gtsam.Pose3):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "state_estimate"
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

    def log_pose(self, pose: gtsam.Pose3, entity: str = "state_estimate"):
        t = pose.translation()
        q = pose.rotation().toQuaternion()
        yaw = pose.rotation().yaw()
        rr.set_time("keyframe", sequence=self.key_count)
        rr.log(
            f"world/{entity}/position",
            rr.Transform3D(
                translation=[t[0], t[1], t[2]],
                rotation=rr.RotationAxisAngle(axis=[0, 0, 1], radians=yaw),
            ),
            static=True,
        )
        rr.log(
            f"world/{entity}/pose",
            rr.Transform3D(
                translation=[t[0], t[1], t[2]],
                rotation=rr.Quaternion(xyzw=[q.x(), q.y(), q.z(), q.w()]),
            ),
            static=True,
        )
        rr.log(
            f"world/{entity}/position/axes",
            rr.TransformAxes3D(axis_length=0.1),
            static=True,
        )
        rr.log(
            f"world/{entity}/pose/axes",
            rr.TransformAxes3D(axis_length=1.0),
            static=True,
        )

        new_point = np.array([[t[0], t[1], t[2]]], dtype=np.float32)
        self.trajectories[entity].points = np.vstack(
            [self.trajectories[entity].points, new_point]
        )

        rr.log(
            f"world/{entity}/trajectory",
            rr.LineStrips3D(
                [self.trajectories[entity].points],
                colors=[self.trajectories[entity].color]
            ),
            static=True,
        )

    def imu_callback(self, msg: SensorCombined):
        # start_time = time.perf_counter()
        # if self.prev_update_timestamp is None:
        #     self.prev_update_timestamp = msg.timestamp
        #     return

        if self.pim is None:
            return

        dt_gyro = msg.gyro_integral_dt * 1e-6  # px4 uses microsec

        gyro = np.array(msg.gyro_rad)
        accel = np.array(msg.accelerometer_m_s2)

        self.pim.integrateMeasurement(accel, gyro, dt_gyro)
        self.imu_meas_count += 1

        if self.imu_meas_count >= self.update_every_n:
            # if msg.timestamp - self.prev_update_timestamp > 50_000:  # 50hz
            graph = gtsam.NonlinearFactorGraph()
            values = gtsam.Values()

            i = self.key_count

            # grab current estimate
            pose = self.isam.calculateEstimatePose3(X(i))
            vel = self.isam.calculateEstimateVector(V(i))
            bias = self.isam.calculateEstimateConstantBias(self.biasKey)
            # self.get_logger().info(f"bias: {bias}")
            prev_state = gtsam.NavState(pose, vel)
            pred_state = self.pim.predict(prev_state, bias)

            pred_pose = pred_state.pose()
            pred_vel = pred_state.velocity()

            self.log_pose(pred_pose, entity="pim")

            if np.any(np.isnan(pred_vel)) or np.any(np.isnan(pred_pose.translation())):
                self.get_logger().warn("predicted state contains NaN, skipping update")
                self.pim.resetIntegration()
                return

            values.insert(X(i + 1), pred_pose)
            values.insert(V(i + 1), pred_vel)

            # add bias factor periodically
            if i % 5 == 0 and i != 0:
                graph.add(
                    gtsam.BetweenFactorConstantBias(
                        self.biasKey,
                        self.biasKey + 1,
                        gtsam.imuBias.ConstantBias(),
                        self.biasNoise,
                    )
                )
                values.insert(
                    self.biasKey + 1,
                    self.isam.calculateEstimateConstantBias(self.biasKey),
                )
                self.biasKey += 1

            if self.latest_gps_msg is not None:
                self.add_gps_factor(graph, X(i + 1), self.latest_gps_msg)
                self.latest_gps_msg = None

            # if self.latest_mag_msg is not None:
            #     self.add_mag_factor(
            #         graph, X(i + 1), self.latest_mag_msg, pred_state.position()
            #     )
                # self.latest_mag_msg = None

            # create imu factor with new bias key
            factor = gtsam.ImuFactor(
                X(i), V(i), X(i + 1), V(i + 1), self.biasKey, self.pim
            )
            graph.add(factor)
            self.pim.resetIntegration()

            # optimize
            self.isam.update(graph, values)
            self.isam.update()  # call twice to help fully relinearize
            pose = self.isam.calculateEstimatePose3(X(i + 1))
            self.latest_pose = pose
            self.log_pose(pose)
            self.publish_pose(pose)

            # if self.key_count % 10 == 0:
            #     self.get_logger().info(str(pose))

            # pose_t = pose.translation()
            # self.get_logger().info(
            #     f"gtsam position: x={pose_t[0]:.2f}, y={pose_t[1]:.2f}, z={pose_t[2]:.2f}"
            # )

            # self.prev_update_timestamp = msg.timestamp
            self.key_count += 1
            self.imu_meas_count = 0

        self.end_time = time.perf_counter()

    def gps_callback(self, msg: SensorGps):
        self.latest_gps_msg = msg

    def magnetometer_callback(self, msg: VehicleMagnetometer):
        self.latest_mag_msg = msg

    def local_position_callback(self, msg: VehicleLocalPosition):
        pass
        # if not hasattr(self, 'latest_pose') or self.latest_pose is None:
        #     return
        #
        # gt_pos = np.array([msg.x, msg.y, msg.z])
        # est_pos = self.latest_pose.translation()
        #
        #
        # pos_error = np.linalg.norm(est_pos - gt_pos)
        #
        # self.get_logger().info(
        #     f"pos error: {pos_error:.3f}m | "
        #     f"gt=({msg.x:.2f}, {msg.y:.2f}, {msg.z:.2f}) | "
        #     f"est=({est_pos[0]:.2f}, {est_pos[1]:.2f}, {est_pos[2]:.2f})"
        # )

    def attitude_callback(self, msg: VehicleAttitude):
        pass
        # if not hasattr(self, "latest_pose") or self.latest_pose is None:
        #     return
        #
        # gt_rot = gtsam.Rot3.Quaternion(msg.q[0], msg.q[1], msg.q[2], msg.q[3])
        # est_rot = self.latest_pose.rotation()
        #
        # # rotation error as angle
        # rot_error = gt_rot.between(est_rot)
        # angle_error = np.degrees(rot_error.axisAngle()[1])
        #
        # # self.get_logger().info(
        # #     f"rot error: {angle_error:.2f}deg | "
        # #     f"gt yaw={np.degrees(gt_rot.yaw()):.1f} | "
        # #     f"est yaw={np.degrees(est_rot.yaw()):.1f}"
        # # )


def main(args=None):
    rclpy.init(args=args)

    state_estimation = StateEstimation()
    rclpy.spin(state_estimation)

    state_estimation.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
