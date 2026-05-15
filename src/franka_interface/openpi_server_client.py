import rospy
from openpi_client import websocket_client_policy
from franka_client import FrankaRobotClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np

HORIZON_LENGTHS = {
    "pi0": 10,
    "pi0_fast": 10,
    "pi05": 15,
}

JOINT_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3",
    "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7",
]

UPDATE_RATE = 10  # Hz

class OpenPiServerClient:
    def __init__(self, host="localhost", port=8000, policy_name="pi05"):
        self.policy_name = policy_name
        self.horizon = HORIZON_LENGTHS.get(policy_name, 15)

        self.franka_client = FrankaRobotClient()

        # Connect to the Pi server via WebSocket
        print(f"Connecting to {self.policy_name} server at {host}:{port}...")
        try:
            self.policy = websocket_client_policy.WebsocketClientPolicy(host, port)
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to {self.policy_name} server at {host}:{port}. "
                f"Make sure the policy server is running.\n{e}"
            )
        print("Connected.")

        self.rate = rospy.Rate(UPDATE_RATE)
        

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def get_images(self):
        """
        Returns (exterior_image, wrist_image) as numpy arrays (H, W, 3) uint8.
        TODO: implement agentlace camera client connection.
        """
        raise NotImplementedError("Camera client not yet implemented")

    def get_joint_positions(self):
        """Returns joint positions as a numpy array of shape (7,)."""
        try:
            joints = self.franka_client.get_current_joints()
            return np.array([joints[j]["position"] for j in JOINT_NAMES])
        except KeyError as e:
            raise RuntimeError(f"Missing expected joint in robot state: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to read joint positions: {e}")

    def get_gripper_position(self):
        """Returns gripper position as a numpy array of shape (1,)."""
        try:
            width = self.franka_client.get_current_gripper_width()["gripper_width"]
            return np.array([width])
        except Exception as e:
            raise RuntimeError(f"Failed to read gripper position: {e}")

    # ------------------------------------------------------------------
    # OpenPi server communication
    # ------------------------------------------------------------------

    def pack_request(self, overhead_image, wrist_image, joint_positions, gripper_position, prompt):
        """Pack the observation data into a format OpenPi server expects."""
        from openpi_client import image_tools
        return {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(overhead_image, 224, 224),
            "observation/wrist_image_left": image_tools.resize_with_pad(wrist_image, 224, 224),
            "observation/joint_position": joint_positions,
            "observation/gripper_position": gripper_position,
            "prompt": prompt
        }

    def infer(self, request):
        """Send the request to the OpenPi server and return the interpreted response."""
        try:
            response = self.policy.infer(request)
        except Exception as e:
            raise RuntimeError(f"Policy server inference failed: {e}")

        if "actions" not in response:
            raise RuntimeError(f"Unexpected response from policy server: {response}")

        actions = np.asarray(response["actions"])

        if actions.shape != (self.horizon, 8):
            raise RuntimeError(
                f"Unexpected action shape from policy server: {actions.shape}, "
                f"expected ({self.horizon}, 8)"
            )

        # Binarize gripper action (last dimension)
        # Mirrors behavior in OpenPi examples
        actions[..., -1] = (actions[..., -1] > 0.5).astype(actions.dtype)
        return actions

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, prompt, num_cycles):
        print(f"Running policy: '{prompt}'")

        cycle = 0
        try:
            while num_cycles is None or cycle < num_cycles:
                # Get image data
                overhead_image, wrist_image = self.get_images()

                # Get proprioceptive data
                joint_positions = self.get_joint_positions()
                gripper_position = self.get_gripper_position()

                # Pack request and run inference
                request = self.pack_request(overhead_image, wrist_image, joint_positions, gripper_position, prompt)
                actions = self.infer(request)

                # Execute action chunk
                prev_gripper = None
                for step in actions:
                    joint_positions = step[:7]
                    gripper_position = step[-1]

                    self.franka_client.publish_joint_positions(joint_positions, JOINT_NAMES, update_rate=UPDATE_RATE)

                    # Only call gripper service if gripper state has changed
                    if prev_gripper is None or gripper_position != prev_gripper:
                        width = 0.085 if gripper_position > 0.5 else 0.0
                        try:
                            self.franka_client.set_gripper_width(width)
                        except Exception as e:
                            print(f"Warning: gripper command failed: {e}")
                        prev_gripper = gripper_position

                    self.rate.sleep()

                cycle += 1

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        except RuntimeError as e:
            print(f"\nRuntime error during policy execution: {e}")
        except Exception as e:
            print(f"\nUnexpected error during policy execution: {e}")
            raise

    def main(self, prompt, num_cycles=None):
        print(f"Running policy with prompt: '{prompt}'")
        print("Press Enter to start, and Ctrl+C to stop at any time.")
        input()

        self.run(prompt, num_cycles)

        print("\nPress 'r' to reset the arm, or any other key to exit.")
        choice = input().strip().lower()
        if choice == 'r':
            print("Resetting robot...")
            try:
                self.franka_client.reset_robot()
                print("Reset complete.")
            except Exception as e:
                print(f"Warning: robot reset failed: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy", default="pi05", choices=["pi0", "pi0_fast", "pi05"])
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--cycles", type=int, default=None)
    args = parser.parse_args()

    try:
        client = OpenPiServerClient(host=args.host, port=args.port, policy_name=args.policy)
        client.main(prompt=args.prompt, num_cycles=args.cycles)
    except RuntimeError as e:
        print(f"Failed to initialize: {e}")
        exit(1)