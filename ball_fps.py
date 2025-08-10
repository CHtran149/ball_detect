import cv2
import torch
import math
import sys
import numpy as np

# Thêm repo yolov5 vào path để import
sys.path.insert(0, '/home/jetson/yolov5')

from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.plots import Annotator, colors

# Hàm scale_coords tự định nghĩa (do repo hiện tại không có sẵn)
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2)  # wh padding
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    return coords

# Cấu hình
MODEL_PATH = "./models/best.pt"
IMG_SIZE = 640
CONF_THRESHOLD = 0.25

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model
model = DetectMultiBackend(MODEL_PATH, device=device)
model.model.eval()

# Mở webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30

point_A = (width // 2, height)  # điểm A: chính giữa đáy khung hình

print("nhan q de thoat")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Khong the lau frame tu webcam")
        break

    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()

    # Chuẩn bị ảnh đầu vào cho model
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img_tensor = torch.from_numpy(img_resized).to(device)
    img_tensor = img_tensor.permute(2, 0, 1).float()  # HWC to CHW
    img_tensor /= 255.0  # Normalize 0-1
    img_tensor = img_tensor.unsqueeze(0)  # Thêm batch dimension

    # Inference
    pred = model(img_tensor)[0]

    # NMS
    pred = non_max_suppression(pred, CONF_THRESHOLD, 0.45)

    ball_count = 0

    annotator = Annotator(frame)

    if pred[0] is not None and len(pred[0]):
        # Scale bbox về kích thước ảnh gốc
        pred_scaled = scale_coords(img_tensor.shape[2:], pred[0][:, :4].clone(), frame.shape).round()

        for i, det in enumerate(pred[0]):
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = pred_scaled[i]

            ball_count += 1
            label = f"Ball {ball_count}"

            # Vẽ bbox
            annotator.box_label((int(x1), int(y1), int(x2), int(y2)), label, color=colors(int(cls), True))

            # Tâm quả bóng
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            ball_center = (center_x, center_y)

            # Tính góc giữa trục dọc và đường nối từ điểm A tới quả bóng
            dx = center_x - point_A[0]
            dy = point_A[1] - center_y  # trục y ngược trong OpenCV

            angle_rad = math.atan2(dx, dy)
            angle_deg = math.degrees(angle_rad)

            # Vẽ đường nối từ A đến tâm bóng
            cv2.line(frame, point_A, ball_center, (0, 0, 255), 2)

            # Ghi nhãn góc lên frame
            cv2.putText(frame, f"{label} ({angle_deg:.1f} deg)", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # In góc ra terminal
            print(f"{label}: angle = {angle_deg:.1f} degrees")

    annotator.text((width - 150, 30), f"Balls: {ball_count}", color=(255, 100, 0), txt_color=(255, 255, 255))
    annotator.apply(frame)

    # Tính FPS
    end_time.record()
    torch.cuda.synchronize()
    fps = 1000 / start_time.elapsed_time(end_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Ball Detection with Angle", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("da thoat chuong trinh.")
