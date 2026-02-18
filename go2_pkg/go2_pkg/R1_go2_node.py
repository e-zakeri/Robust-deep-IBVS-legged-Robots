#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from geometry_msgs.msg import Vector3
from cv_bridge import CvBridge
import cv2
import asyncio
import threading
import json
from queue import Queue

from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD
from aiortc import MediaStreamTrack

import numpy as np
# ========================= CONFIGURATION =========================

# --- Node name ---
NODE_NAME = "R1_go2_node"

# --- Topics ---
CAMERA_TOPIC       = "/camera/image_raw"
CMD_SPEED_TOPIC    = "/cmd/speed_level"
CMD_MOVE_TOPIC     = "/cmd/move"
CMD_EULER_TOPIC    = "/cmd/euler"
CMD_HEIGHT_TOPIC   = "/cmd/body_height"

# --- Publishing rate ---
PUBLISH_RATE_HZ    = 30.0    # camera publish rate

# --- Fisheye undistortion ---
USE_UNDISTORT      = True    # set False to disable undistortion
UNDISTORT_BALANCE  = 0.0     # [0..1], trade-off FOV vs cropping

# Calibrated camera matrix
CAMERA_MATRIX_K = np.array([
    [8.04269690e+02, 6.55249202e-02, 6.52582186e+02],
    [0.0,            8.06897068e+02, 3.71434711e+02],
    [0.0,            0.0,            1.0]
], dtype=np.float64)

# Fisheye distortion coefficients
DIST_COEFFS_D = np.array([
    -0.14184394,
    0.56786194,
    -2.06366362,
    2.37330937
], dtype=np.float64)

# --- WebRTC connection mode ---
# You can switch to LocalSTA + IP if needed:
#   WEBRTC_METHOD = WebRTCConnectionMethod.LocalSTA
#   WEBRTC_IP     = "10.42.0.120"
WEBRTC_METHOD = WebRTCConnectionMethod.LocalAP
WEBRTC_IP     = None   # only used for LocalSTA

# ================================================================


class Go2ROS2Node(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        self.bridge = CvBridge()

        # ---- Publishers & Subscribers ----
        self.image_pub = self.create_publisher(Image, CAMERA_TOPIC, 1)
        self.create_subscription(Float32, CMD_SPEED_TOPIC,  self.speed_level_callback, 1)
        self.create_subscription(Vector3, CMD_MOVE_TOPIC,   self.move_callback,        1)
        self.create_subscription(Vector3, CMD_EULER_TOPIC,  self.euler_callback,       1)
        self.create_subscription(Float32, CMD_HEIGHT_TOPIC, self.body_height_callback, 1)

        # ---- Frame queue & WebRTC connection ----
        self.frame_queue = Queue()

        if WEBRTC_METHOD == WebRTCConnectionMethod.LocalSTA and WEBRTC_IP is not None:
            self.conn = Go2WebRTCConnection(WEBRTC_METHOD, ip=WEBRTC_IP)
        else:
            self.conn = Go2WebRTCConnection(WEBRTC_METHOD)

        # ---- Undistortion config / cache ----
        self.use_undistort       = USE_UNDISTORT
        self._undistort_map1     = None
        self._undistort_map2     = None
        self._undistort_dim      = None   # (w, h)
        self._undistort_balance  = None   # last used balance

        # ---- AsyncIO loop in a thread ----
        self.loop = asyncio.new_event_loop()
        self.async_thread = threading.Thread(
            target=self._run_async_loop,
            args=(self.loop,),
            daemon=True,
        )
        self.async_thread.start()

        # ---- Timer to publish at PUBLISH_RATE_HZ ----
        self.timer = self.create_timer(1.0 / PUBLISH_RATE_HZ, self.publish_frame)

        self.get_logger().info(
            f"Go2ROS2Node initialized. "
            f"Publishing camera on {CAMERA_TOPIC} at {PUBLISH_RATE_HZ:.1f} Hz, "
            f"undistort: {'ON' if self.use_undistort else 'OFF'}."
        )

    # ----------------------------------------------------------------------
    #                      ASYNCIO / WEBRTC SETUP
    # ----------------------------------------------------------------------
    def _run_async_loop(self, loop):
        """Start asyncio loop and set up WebRTC connection."""
        asyncio.set_event_loop(loop)

        async def setup():
            try:
                await self.conn.connect()
                self.conn.video.switchVideoChannel(True)
                await self._set_motion_mode("normal")
                self.conn.video.add_track_callback(self.recv_camera_stream)
                self.get_logger().info("WebRTC connected, video on, motion mode=normal")
            except Exception as e:
                self.get_logger().error(f"WebRTC setup error: {e}")

        loop.run_until_complete(setup())
        loop.run_forever()

    async def recv_camera_stream(self, track: MediaStreamTrack):
        """Receive frames from WebRTC and enqueue them."""
        while rclpy.ok():
            try:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                self.frame_queue.put(img)
            except Exception as e:
                self.get_logger().error(f"Error in recv_camera_stream: {e}")
            await asyncio.sleep(0.001)

    # ----------------------------------------------------------------------
    #                         IMAGE PUBLISHING
    # ----------------------------------------------------------------------
    def publish_frame(self):
        """Publish the latest frame from the queue."""
        if not self.frame_queue.empty():
            img = self.frame_queue.get()
            try:
                if self.use_undistort:
                    img = self.undistort_fisheye(img, balance=UNDISTORT_BALANCE)

                ros_img = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
                ros_img.header.stamp = self.get_clock().now().to_msg()
                ros_img.header.frame_id = "camera_frame"
                self.image_pub.publish(ros_img)
            except Exception as e:
                self.get_logger().error(f"Error publishing image: {e}")

    # ----------------------------------------------------------------------
    #                      FISHEYE UNDISTORTION
    # ----------------------------------------------------------------------
    def undistort_fisheye(self, img: np.ndarray, balance: float = 0.0) -> np.ndarray:
        """
        Undistort a fisheye image using hard-coded camera parameters.
        The rectification maps are cached and reused for performance.

        Args:
            img:     Input distorted image (H×W×C or H×W).
            balance: Trade-off between FOV and cropping [0: tight crop → 1: keep all FOV].

        Returns:
            Undistorted image, same dtype & size as input.
        """
        h, w = img.shape[:2]
        dim = (w, h)

        # Recompute maps only if:
        #  - first time, or
        #  - resolution changed, or
        #  - balance changed
        if (
            self._undistort_map1 is None
            or self._undistort_dim != dim
            or self._undistort_balance is None
            or abs(self._undistort_balance - balance) > 1e-6
        ):
            self.get_logger().info(
                f"Computing fisheye undistort maps for resolution {w}x{h}, "
                f"balance={balance}"
            )

            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                CAMERA_MATRIX_K, DIST_COEFFS_D, dim, np.eye(3), balance=balance
            )

            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                CAMERA_MATRIX_K, DIST_COEFFS_D, np.eye(3), new_K, dim, cv2.CV_16SC2
            )

            self._undistort_map1 = map1
            self._undistort_map2 = map2
            self._undistort_dim = dim
            self._undistort_balance = balance

        # Fast remap using cached maps
        undistorted = cv2.remap(
            img,
            self._undistort_map1,
            self._undistort_map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        return undistorted

    # ----------------------------------------------------------------------
    #                           COMMAND CALLBACKS
    # ----------------------------------------------------------------------
    def speed_level_callback(self, msg: Float32):
        asyncio.run_coroutine_threadsafe(self._set_speed_level(msg.data), self.loop)

    async def _set_speed_level(self, level: float):
        await self.conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": SPORT_CMD["SpeedLevel"], "parameter": {"data": level}},
        )

    def move_callback(self, msg: Vector3):
        asyncio.run_coroutine_threadsafe(
            self._move_command(msg.x, msg.y, msg.z),
            self.loop,
        )

    async def _move_command(self, x: float, y: float, z: float):
        await self.conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": SPORT_CMD["Move"], "parameter": {"x": x, "y": y, "z": z}},
        )

    def euler_callback(self, msg: Vector3):
        asyncio.run_coroutine_threadsafe(
            self._euler_command(msg.x, msg.y, msg.z),
            self.loop,
        )

    async def _euler_command(self, x: float, y: float, z: float):
        await self.conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": SPORT_CMD["Euler"], "parameter": {"x": x, "y": y, "z": z}},
        )

    def body_height_callback(self, msg: Float32):
        asyncio.run_coroutine_threadsafe(
            self._body_height_command(msg.data),
            self.loop,
        )

    async def _body_height_command(self, height: float):
        await self.conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": SPORT_CMD["BodyHeight"], "parameter": {"data": height}},
        )

    async def _set_motion_mode(self, mode: str):
        # Query current mode
        resp = await self.conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"], {"api_id": 1001}
        )
        try:
            current = json.loads(resp["data"]["data"]).get("name", "")
        except Exception:
            current = ""
        if current != mode:
            await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"],
                {"api_id": 1002, "parameter": {"name": mode}},
            )
            self.get_logger().info(f"Switched motion mode to {mode}")

    # ----------------------------------------------------------------------
    #                           SHUTDOWN
    # ----------------------------------------------------------------------
    def shutdown(self):
        """Clean shutdown: stop timer, stop loop, join thread."""
        self.get_logger().info("Shutting down Go2ROS2Node")
        self.timer.cancel()
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            pass
        try:
            self.async_thread.join(timeout=2.0)
        except Exception:
            pass


# ============================== MAIN ===============================

def main(args=None):
    rclpy.init(args=args)
    node = Go2ROS2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
