import torch
import cv2
import time
import math

import warnings
warnings.filterwarnings("ignore")

# ==== CẤU HÌNH ====
MODEL_PATH = "yolov5/runs/train/roboflow_yolov52/weights/best.pt"  # Đường dẫn tới model bạn đã huấn luyện
VIDEO_PATH = "output (1).mp4"
OUTPUT_PATH = "output_detected_2.mp4"
IMG_SIZE = 640
CONF_THRESHOLD = 0.25

# ==== Load model YOLOv5 ====
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
model.conf = CONF_THRESHOLD
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ==== Mở video ====
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video = cap.get(cv2.CAP_PROP_FPS)

# Ghi video đầu ra
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_video, (width, height))

# ==== Xử lý từng frame ====
print("🚀 Đang xử lý video...")
prev_time = time.time()

# Điểm A: chính giữa đáy khung hình
point_A = (width // 2, height)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
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

        # Vẽ khung
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Tâm quả bóng
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        ball_center = (center_x, center_y)

        # Tính góc giữa trục dọc và đường nối từ điểm A tới quả bóng
        dx = center_x - point_A[0]
        dy = point_A[1] - center_y  # trục y ngược trong OpenCV

        angle_rad = math.atan2(dx, dy)  # dx trước vì trục gốc là dọc
        angle_deg = math.degrees(angle_rad)

        # Vẽ đường nối từ A đến tâm bóng
        cv2.line(frame, point_A, ball_center, (0, 0, 255), 2)

        # Ghi nhãn bóng và góc
        cv2.putText(frame, f"{label} ({angle_deg:.1f} deg)", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Hiển thị tổng số bóng góc trên phải
    cv2.putText(frame, f"Balls: {ball_count}", (width - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 0), 2)

    # Tính FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time + 1e-6)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Ghi và hiển thị
    out.write(frame)
    cv2.imshow("Ball Detection with Angle", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Video đã lưu tại: {OUTPUT_PATH}")