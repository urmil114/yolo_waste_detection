import cv2
import numpy as np
import time
import datetime
import platform
import sys
import csv
from collections import defaultdict
from ultralytics import YOLO
import onnxruntime as ort

# ================================
# 1. Cross-platform Alert System
# ================================
if platform.system() == "Windows":
    import winsound
    def alert(freq, duration):
        winsound.Beep(freq, duration)
else:
    def alert(freq, duration):
        sys.stdout.write("\a")
        sys.stdout.flush()

# ================================
# 2. Load YOLO Model (detection)
# ================================
yolo_model = YOLO("runs/detect/train/weights/best.pt")

# ================================
# 3. Load EfficientNet ONNX (classification)
# ================================
onnx_model_path = "efficientnet/effnet_b2.onnx"
session = ort.InferenceSession(
    onnx_model_path,
    providers=["CPUExecutionProvider"]  # change to ["CUDAExecutionProvider"] if GPU is available
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

expected_h, expected_w = session.get_inputs()[0].shape[2:4]
effnet_size = expected_h if isinstance(expected_h, int) else 224

# Class labels
class_labels = ["hazardous", "organic", "recyclable"]

# Colors for drawing
colors = {
    "hazardous": (0, 0, 255),
    "organic": (0, 255, 0),
    "recyclable": (255, 165, 0)
}

# ================================
# 4. Preprocess for EfficientNet
# ================================
def preprocess(image, size=effnet_size):
    """Resize and normalize image for EfficientNet-B2 (ImageNet normalization)."""
    image = cv2.resize(image, (size, size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

# ================================
# 5. Process Frame
# ================================
def process_frame(frame, class_counts, conf_thresh=0.5):
    detections = []
    results = yolo_model(frame, stream=True)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)

        for box, yolo_conf, cls in zip(boxes, confs, clss):
            x1, y1, x2, y2 = box
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # Classification with EfficientNet
            input_tensor = preprocess(roi)
            pred = session.run([output_name], {input_name: input_tensor})[0]
            class_id = int(np.argmax(pred))
            effnet_conf = float(np.max(pred))
            label = class_labels[class_id]

            if effnet_conf >= conf_thresh:
                class_counts[label] += 1
                detections.append((label, float(yolo_conf), effnet_conf))
                color = colors[label]

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label} | YOLO:{yolo_conf:.2f}, Eff:{effnet_conf:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
                cv2.putText(frame, text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame, detections

# ================================
# 6. Main Inference Loop
# ================================
def run(video_source=0, save_output=False, output_path=None, log_csv=True):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"❌ Could not open video source {video_source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 20.0

    writer = None
    recording = save_output
    if save_output:
        if not output_path:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"output_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 Saving video to {output_path}")

    # CSV logging
    csv_file, csv_writer = None, None
    if log_csv:
        csv_file = open("detections_log.csv", mode="w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Frame", "Label", "YOLO_Conf", "EffNet_Conf"])

    prev_time = time.time()
    fps_smooth = fps
    frame_id = 0

    print("✅ Controls: [q]=quit, [s]=screenshot, [r]=record toggle")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame grab failed, retrying...")
            time.sleep(0.5)
            continue

        frame_id += 1
        class_counts = defaultdict(int)
        frame, detections = process_frame(frame, class_counts)

        # Alerts + Logging
        for label, yolo_conf, effnet_conf in detections:
            if csv_writer:
                csv_writer.writerow([frame_id, label, yolo_conf, effnet_conf])

            if label == "hazardous" and effnet_conf > 0.75:
                alert(1000, 400)
                print("⚠️ Hazardous object detected!")

            elif label == "organic" and effnet_conf > 0.75:
                alert(600, 200)
                print("🌱 Organic waste detected.")

            elif label == "recyclable" and effnet_conf > 0.75:
                alert(800, 200)
                print("♻️ Recyclable detected.")

        # FPS calculation
        curr_time = time.time()
        fps_val = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_smooth = (fps_smooth * 0.9) + (fps_val * 0.1)

        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Class counts display
        y_offset = 70
        for cls, count in class_counts.items():
            cv2.putText(frame, f"{cls}: {count}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[cls], 2)
            y_offset += 30

        cv2.imshow("YOLO + EfficientNet Inference", frame)

        if recording and writer:
            writer.write(frame)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            img_name = f"screenshot_{ts}.jpg"
            cv2.imwrite(img_name, frame)
            print(f"📸 Screenshot saved: {img_name}")
        elif key == ord("r"):
            recording = not recording
            print("▶️ Recording" if recording else "⏸️ Recording stopped")

    cap.release()
    if writer: writer.release()
    if csv_file: csv_file.close()
    cv2.destroyAllWindows()

# ================================
# 7. Run
# ================================
if __name__ == "__main__":
    run(0, save_output=True, output_path="advanced_output.mp4", log_csv=True)
