"""Live ROS1 service client for the Franka Panda + Robotiq 2F-85 robot API.

Each public method corresponds to one entry in FrankaConfig.api_spec and is
backed by a ROS1 service call using the same custom srv types as the Stretch:

    RobotQuery   — no request fields; response: {result_code, data: JSON string}
    RobotCommand — request: {req: JSON string}; response: {result_code, data}

Service name pattern: /robot/{category}/{function_name}

ROS environment: ROS1 Noetic (rospy), running on the Franka control PC (joeljang).
The SSA process must source the catkin workspace before importing this module:
    source /home/daphne/archit/ssa_ws/devel/setup.bash

TODO: confirm the package that provides RobotCommand / RobotQuery srv types and
update the import below. On the Stretch the package is `robot_api_interfaces`;
on the Franka it may be `franka_robot_apis` or the same shared package.
"""

from __future__ import annotations

import json

# TODO: update this import once the srv package name is confirmed on joeljang.
# Run `rosservice type /robot/control/move_ee_to_pose` to find the package.
try:
    import rospy
    from robot_api_interfaces.srv import (  # type: ignore[import]
        RobotCommand,
        RobotCommandRequest,
        RobotQuery,
        RobotQueryRequest,
    )
    _ROS_AVAILABLE = True
except ImportError:
    _ROS_AVAILABLE = False


class FrankaRobotClient:
    """ROS1 client that exposes the Franka robot API as Python method calls.

    Service proxies are created lazily and cached so we don't reconnect on
    every call. All methods are synchronous (rospy service calls block until
    the service responds).
    """

    def __init__(self) -> None:
        if not _ROS_AVAILABLE:
            raise RuntimeError(
                "rospy / robot_api_interfaces not available. "
                "Source the catkin workspace before running in live mode."
            )
        rospy.init_node("ssa_franka_client", anonymous=True, disable_signals=True)
        self._query_proxies: dict[str, rospy.ServiceProxy] = {}
        self._command_proxies: dict[str, rospy.ServiceProxy] = {}

        self.self.joint_pub = rospy.Publisher(
            "/position_joint_trajectory_controller/command",
            JointTrajectory,
            queue_size=1,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query(self, service_name: str) -> dict:
        """Call a RobotQuery service (no input) and return parsed JSON data."""
        if service_name not in self._query_proxies:
            rospy.wait_for_service(service_name)
            self._query_proxies[service_name] = rospy.ServiceProxy(
                service_name, RobotQuery
            )
        resp = self._query_proxies[service_name](RobotQueryRequest())
        if resp.result_code.result_code != 0:
            raise RuntimeError(resp.result_code.message)
        return json.loads(resp.data)

    def _command(self, service_name: str, payload: dict) -> dict:
        """Call a RobotCommand service with a JSON payload and return parsed data."""
        if service_name not in self._command_proxies:
            rospy.wait_for_service(service_name)
            self._command_proxies[service_name] = rospy.ServiceProxy(
                service_name, RobotCommand
            )
        req = RobotCommandRequest(req=json.dumps(payload))
        resp = self._command_proxies[service_name](req)
        if resp.result_code.result_code != 0:
            raise RuntimeError(resp.result_code.message)
        return json.loads(resp.data)

    def destroy_node(self) -> None:
        """Shut down the ROS1 node."""
        rospy.signal_shutdown("SSA session ended")

    # ------------------------------------------------------------------
    # Proprioception
    # ------------------------------------------------------------------

    def get_current_joints(self) -> dict:
        return self._query("/robot/proprioception/get_current_joints")["joints"]

    def get_current_gripper_width(self) -> dict:
        return self._query("/robot/proprioception/get_gripper_width")

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def set_gripper_width(self, width: float) -> bool:
        return self._command(
            "/robot/control/set_gripper_width", {"width": width}
        )["success"]

    def publish_joint_positions(self, joint_positions, joint_names, update_rate=10):
        msg = JointTrajectory()
        msg.joint_names = joint_names
        point = JointTrajectoryPoint()
        point.positions = joint_positions.tolist()
        point.time_from_start = rospy.Duration(1 / update_rate)
        msg.points = [point]
        self.joint_pub.publish(msg)

    def reset_robot(self) -> bool:
        return self._query("/robot/control/reset_robot")["success"]
