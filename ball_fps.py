import torch
import cv2
import time
import math

import warnings
warnings.filterwarnings("ignore")

# ==== CAU HINH ====
MODEL_PATH = "yolov5/runs/train/roboflow_yolov52/weights/best.pt"  # Duong dan toi model ban da train
CAM_INDEX = 0
IMG_SIZE = 640
CONF_THRESHOLD = 0.25

# ==== Load model YOLOv5 ====
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
model.conf = CONF_THRESHOLD
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ==== Mo webcam ====
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print(f"Error: Khong mo duoc webcam voi index {CAM_INDEX}")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Diem A: chinh giua day khung hinh
point_A = (width // 2, height)

print("🚀 Dang quet webcam va phat hien bong... Nhan Ctrl+C de dung.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Loi: Khong lay duoc frame tu webcam.")
            break

        start_time = time.time()

        # Inference YOLOv5
        results = model(frame, size=IMG_SIZE)
        detections = results.xyxy[0].cpu().numpy()  # (x1, y1, x2, y2, conf, class)
        class_names = results.names

        ball_count = 0

        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls_id = det
            class_name = class_names[int(cls_id)]

            ball_count += 1
            label = f"Ball {ball_count}"

            # Tam qua bong
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Tinh goc giua truc doc va duong noi tu diem A toi qua bong
            dx = center_x - point_A[0]
            dy = point_A[1] - center_y  # truc y nguoc trong OpenCV

            angle_rad = math.atan2(dx, dy)  # dx truoc vi truc goc la doc
            angle_deg = math.degrees(angle_rad)

            print(f"{label}: Goc = {angle_deg:.1f} do")

        # Tinh FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time + 1e-6)
        print(f"Phat hien tong so bong: {ball_count} | FPS: {fps:.2f}\n")

except KeyboardInterrupt:
    print("Da dung quet webcam.")

finally:
    cap.release()
