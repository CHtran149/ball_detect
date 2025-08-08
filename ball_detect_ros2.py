#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import torch
import math
import time
import numpy as np
from threading import Lock

# ====== CẤU HÌNH ======
MODEL_PATH = "/path/to/your/best.pt"   # <- đổi đường dẫn tới model của bạn
CAM_INDEX = 0                          # webcam index
IMG_SIZE = 640
CONF_THRESHOLD = 0.25

# PID params (tune cho robot của bạn)
KP_ANG = 1.2
KI_ANG = 0.0
KD_ANG = 0.05
KP_LIN = 0.6
KI_LIN = 0.0
KD_LIN = 0.02

MAX_LINEAR = 0.5   # m/s
MAX_ANGULAR = 1.5  # rad/s

# Thresholds
APPROACH_BBOX_HEIGHT_THRESHOLD = 180   # nếu bbox cao hơn => đã tới gần (tùy camera)
RETURN_TOL = 0.05                       # m, khoảng cách chấp nhận khi quay về vị trí gốc

# State machine
STATE_IDLE = "IDLE"
STATE_APPROACH = "APPROACH"
STATE_RETURN = "RETURN"

class PID:
    def __init__(self, kp, ki, kd, mn=-1e9, mx=1e9):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.mn = mn; self.mx = mx
        self.integral = 0.0
        self.prev = None
        self.prev_time = None

    def reset(self):
        self.integral = 0.0
        self.prev = None
        self.prev_time = None

    def __call__(self, error):
        t = time.time()
        dt = 1e-6 if self.prev_time is None else max(1e-6, t - self.prev_time)
        self.prev_time = t
        self.integral += error * dt
        deriv = 0.0 if self.prev is None else (error - self.prev) / dt
        self.prev = error
        out = self.kp * error + self.ki * self.integral + self.kd * deriv
        return max(self.mn, min(self.mx, out))

class BallFollower(Node):
    def __init__(self):
        super().__init__('ball_follower')
        self.declare_parameter('model_path', MODEL_PATH)

        # Load model
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.get_logger().info(f"Loading YOLOv5 model from {model_path} ...")
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        self.model.conf = CONF_THRESHOLD
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.get_logger().info(f"Model loaded. Using device: {device}")

        # ROS publishers/subscribers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)

        # Webcam
        import cv2
        self.cap = cv2.VideoCapture(CAM_INDEX)
        if not self.cap.isOpened():
            self.get_logger().error(f"Cannot open webcam index {CAM_INDEX}")
            raise RuntimeError("Webcam not available")

        # State & data
        self.state = STATE_IDLE
        self.lock = Lock()
        self.current_odom = None
        self.home_pose = None
        self.current_target_bbox = None
        self.targets_handled = []  # các bóng đã xử lý

        # PID controllers
        self.pid_ang = PID(KP_ANG, KI_ANG, KD_ANG, mn=-MAX_ANGULAR, mx=MAX_ANGULAR)
        self.pid_lin = PID(KP_LIN, KI_LIN, KD_LIN, mn=-MAX_LINEAR, mx=MAX_LINEAR)

        # Timer callback
        self.timer = self.create_timer(0.05, self.timer_cb)
        self.get_logger().info("Ball follower node started.")

    def odom_callback(self, msg: Odometry):
        with self.lock:
            px = msg.pose.pose.position.x
            py = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            self.current_odom = (px, py, yaw)

    def timer_cb(self):
        import cv2
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn_once("Frame read failed from webcam.")
            return

        h, w = frame.shape[:2]
        # YOLOv5 detect
        results = self.model(frame, size=IMG_SIZE)
        detections = results.xyxy[0].cpu().numpy()

        balls = []
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            bbox_h = int(y2 - y1)
            balls.append({'cx': cx, 'cy': cy, 'bbox_h': bbox_h})

        balls_sorted = sorted(balls, key=lambda b: b['cx'])
        balls_to_consider = [
            b for b in balls_sorted
            if all(abs(b['cx'] - hh[0]) >= 30 or abs(b['cy'] - hh[1]) >= 30
                   for hh in self.targets_handled)
        ]

        # FSM
        if self.state == STATE_IDLE:
            if balls_to_consider:
                self.current_target_bbox = balls_to_consider[0]
                with self.lock:
                    self.home_pose = self.current_odom
                self.pid_ang.reset(); self.pid_lin.reset()
                self.state = STATE_APPROACH
                self.get_logger().info(f"State -> APPROACH")
            else:
                self.publish_twist(0.0, 0.0)

        elif self.state == STATE_APPROACH:
            if not self.current_target_bbox:
                self.state = STATE_IDLE
            else:
                t = self.current_target_bbox
                img_cx = w // 2
                img_cy = h
                dx = t['cx'] - img_cx
                dy = img_cy - t['cy']
                angle_rad = math.atan2(dx, dy)
                dist_error = (APPROACH_BBOX_HEIGHT_THRESHOLD - t['bbox_h']) / max(1.0, t['bbox_h'])
                w_cmd = - self.pid_ang(angle_rad)
                v_cmd = self.pid_lin(dist_error)
                v_cmd = max(-MAX_LINEAR, min(MAX_LINEAR, v_cmd))
                w_cmd = max(-MAX_ANGULAR, min(MAX_ANGULAR, w_cmd))

                if t['bbox_h'] >= APPROACH_BBOX_HEIGHT_THRESHOLD:
                    self.publish_twist(0.0, 0.0)
                    self.targets_handled.append((t['cx'], t['cy']))
                    self.state = STATE_RETURN
                    time.sleep(0.5)
                    return
                else:
                    self.publish_twist(v_cmd, w_cmd)
                    if balls_to_consider:
                        nearest = min(balls_to_consider, key=lambda b: abs(b['cx'] - t['cx']))
                        self.current_target_bbox = nearest

        elif self.state == STATE_RETURN:
            with self.lock:
                home = self.home_pose
                od = self.current_odom
            if home is None or od is None:
                self.state = STATE_IDLE
                return
            dx = home[0] - od[0]
            dy = home[1] - od[1]
            dist = math.hypot(dx, dy)
            angle_to_home = math.atan2(dy, dx)
            yaw_err = angle_normalize(angle_to_home - od[2])

            if abs(yaw_err) > 0.15:
                w_cmd = max(-MAX_ANGULAR, min(MAX_ANGULAR, 1.2 * yaw_err))
                self.publish_twist(0.0, w_cmd)
            elif dist > RETURN_TOL:
                v_cmd = max(-MAX_LINEAR, min(MAX_LINEAR, 0.6 * dist))
                w_cmd = max(-MAX_ANGULAR, min(MAX_ANGULAR, 0.8 * yaw_err))
                self.publish_twist(v_cmd, w_cmd)
            else:
                self.publish_twist(0.0, 0.0)
                self.current_target_bbox = None
                self.state = STATE_IDLE

    def publish_twist(self, v, w):
        t = Twist()
        t.linear.x = float(v)
        t.angular.z = float(w)
        self.cmd_pub.publish(t)

def angle_normalize(x):
    return (x + math.pi) % (2*math.pi) - math.pi

def main(args=None):
    rclpy.init(args=args)
    node = BallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
