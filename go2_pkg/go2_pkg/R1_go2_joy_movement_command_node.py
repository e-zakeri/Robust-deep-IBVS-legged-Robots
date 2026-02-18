#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Joy
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32


# ====================== CONFIGURATION PARAMETERS ======================
#
# Button indices (0-based)
BTN_DEADMAN = 6
BTN_MODE    = 0
BTN_UP      = 2
BTN_DN      = 4
#
# Throttle axis index
THROTTLE_AXIS = 3 # check to see the range. It must be between 0 and 1
THROTTLE_ENABLE = True
#
# Independent scaling for all 6 DoF
#  [Tx, Ty, Tz, Rx, Ry, Rz]
#
# Values chosen to match original behavior:
#   movement: no scaling
#   rotation: 0.5
#
VEL_SCALE = [
    1.0,  # Tx: translation x
    1.0,  # Ty: translation y
    1.0,  # Tz: translation z
    0.5,  # Rx: roll
    0.5,  # Ry: pitch
    0.5,  # Rz: yaw
]
#
# Body height control
STEP_H       = 0.01
MIN_H, MAX_H = -0.1, 0.05
#
# Debug
DEBUG_HEIGHT = False
DEBUG_SCALE  = False
# =====================================================================


class JoyToMoveNode(Node):
    """
    Joystick -> robot command node.

    Movement and rotation scaling:
      effective_output = VEL_SCALE[i] * throttle
    """

    def __init__(self):
        super().__init__("R1_go2_joy_movement_command_node")

        # Publishers
        self.pub_cmd = self.create_publisher(Vector3, "/cmd/move", 10)
        self.pub_cmd_E = self.create_publisher(Vector3, "/cmd/euler", 10)
        self.pub_height = self.create_publisher(Float32, "/cmd/body_height", 10)

        # Subscriber
        self.create_subscription(Joy, "/joy", self.joy_callback, 10)

        # Internal state
        self.cur_height = 0.0
        self.prev_btns = []

        self.get_logger().info(
            f"Joystick node started. Hold button #{BTN_DEADMAN+1} to enable motion."
        )

    # ----------------------------------------------------------------------
    #                             CALLBACK
    # ----------------------------------------------------------------------
    def joy_callback(self, msg: Joy):

        # First joystick event: record button states
        if not self.prev_btns:
            self.prev_btns = list(msg.buttons)

        # ==================================================================
        # HEIGHT CONTROL
        # ==================================================================
        if msg.buttons[BTN_UP] and not self.prev_btns[BTN_UP]:
            self.adjust_height(+STEP_H)

        if msg.buttons[BTN_DN] and not self.prev_btns[BTN_DN]:
            self.adjust_height(-STEP_H)

        self.prev_btns = list(msg.buttons)  # update

        # ==================================================================
        # MOTION CONTROL
        # ==================================================================
        if msg.buttons[BTN_DEADMAN] != 1:
            return

        scale = self.compute_throttle_scale(msg)

        # ======================
        # MODE 0: translation
        # ======================
        if msg.buttons[BTN_MODE] == 0:

            x = msg.axes[1] * VEL_SCALE[0] * scale
            y = msg.axes[0] * VEL_SCALE[1] * scale
            z = msg.axes[2] * VEL_SCALE[2] * scale

            self.pub_cmd.publish(Vector3(x=x, y=y, z=z))

        # ======================
        # MODE 1: euler rotation
        # ======================
        else:
            rx = -msg.axes[0] * VEL_SCALE[3] * scale
            ry =  msg.axes[1] * VEL_SCALE[4] * scale
            rz =  msg.axes[2] * VEL_SCALE[5] * scale

            self.pub_cmd_E.publish(Vector3(x=rx, y=ry, z=rz))

    # ----------------------------------------------------------------------
    #                     HELPER: COMPUTE THROTTLE SCALE
    # ----------------------------------------------------------------------
    def compute_throttle_scale(self, msg: Joy) -> float:

        if not THROTTLE_ENABLE:
            return 1.0

        if THROTTLE_AXIS >= len(msg.axes):
            self.get_logger().warn(
                f"Throttle axis {THROTTLE_AXIS} out of range; using 1.0"
            )
            return 1.0

        raw = msg.axes[THROTTLE_AXIS]

        # normalize raw → [0..1]
        #if raw < 0 or raw > 1:
        throttle = (raw + 1.0) * 0.5
        #else:
        #    throttle = raw

        throttle = max(0.0, min(1.0, throttle))

        if DEBUG_SCALE:
            self.get_logger().info(
                f"Throttle raw={raw:.2f}, scaled={throttle:.2f}"
            )

        return throttle

    # ----------------------------------------------------------------------
    #                         HELPER: HEIGHT
    # ----------------------------------------------------------------------
    def adjust_height(self, delta: float):

        self.cur_height = max(MIN_H, min(MAX_H, self.cur_height + delta))
        self.pub_height.publish(Float32(data=self.cur_height))

        if DEBUG_HEIGHT:
            self.get_logger().info(f"Body height set to {self.cur_height:.3f} m")


# ----------------------------------------------------------------------
#                               MAIN
# ----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = JoyToMoveNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
