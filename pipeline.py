import cv2
import numpy as np
import time
from ultralytics import YOLO
import onnxruntime as ort
import csv

# ================================
# 1. Load YOLO model (detection)
# ================================
yolo_model = YOLO("runs/detect/train/weights/best.pt")  # trained YOLO model

# ================================
# 2. Load EfficientNet ONNX (classification)
# ================================
onnx_model_path = "efficientnet/effnet_b2.onnx"
session = ort.InferenceSession(
    onnx_model_path,
    providers=["CPUExecutionProvider"] 
    # change to ["CUDAExecutionProvider"] if GPU is available
)
#output na,e
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Detect correct input size from ONNX model
expected_h, expected_w = session.get_inputs()[0].shape[2:4]
effnet_size = expected_h if isinstance(expected_h, int) else 224

# Class labels (must match training order)
class_labels = ["hazardous", "organic", "recyclable"]

# Class-specific colors (BGR format)
colors = {
    "hazardous": (0, 0, 255),      # Red
    "organic": (0, 255, 0),        # Green
    "recyclable": (255, 165, 0)    # Orange
}

# ================================
# 3. Preprocess for EfficientNet
# ================================
def preprocess(image, size=effnet_size):
    """Resize and normalize image for EfficientNet-B2 (ImageNet normalization)."""
    image = cv2.resize(image, (size, size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    # 🔑 Proper ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std

    image = np.transpose(image, (2, 0, 1))  # HWC → CHW
    image = np.expand_dims(image, axis=0)   # Add batch dim
    return image

# ================================
# 4. Frame Processing Function
# ================================
def process_frame(frame, class_counts, conf_thresh=0.5):
    detections = []

    # Run YOLO detection
    results = yolo_model(frame, stream=True)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)   # Bounding boxes
        confs = r.boxes.conf.cpu().numpy()               # YOLO confidence
        clss = r.boxes.cls.cpu().numpy().astype(int)     # YOLO class IDs

        for box, yolo_conf, cls in zip(boxes, confs, clss):
            x1, y1, x2, y2 = box
            roi = frame[y1:y2, x1:x2]  # crop detected region

            if roi.size == 0:
                continue

            # Preprocess for EfficientNet
            input_tensor = preprocess(roi)

            # Run classification
            pred = session.run([output_name], {input_name: input_tensor})[0]
            class_id = int(np.argmax(pred))
            effnet_conf = float(np.max(pred))
            label = class_labels[class_id]

            if effnet_conf >= conf_thresh:
                # update counts
                class_counts[label] += 1

                # Save detection
                detections.append((label, float(yolo_conf), effnet_conf))

                # Choose color by class
                color = colors[label]

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Put label + both confidences
                text = f"{label} | YOLO:{yolo_conf:.2f}, Eff:{effnet_conf:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)  # background
                cv2.putText(frame, text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame, detections

# ================================
# 5. Run Pipeline (Webcam/Video)
# ================================
def run_pipeline(video_source=0, save_output=False, output_path="output.mp4", log_csv=False):
    cap = cv2.VideoCapture(video_source)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0

    # Video writer (optional)
    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # CSV logging
    csv_file = None
    csv_writer = None
    if log_csv:
        csv_file = open("detections_log.csv", mode="w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Frame", "Label", "YOLO_Conf", "EffNet_Conf"])

    prev_time = time.time()
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # ✅ Reset counts per frame
        class_counts = {"hazardous": 0, "organic": 0, "recyclable": 0}

        # Run processing (with class counts)
        frame, detections = process_frame(frame, class_counts)

        # Log detections
        if log_csv and csv_writer:
            for (label, yolo_conf, effnet_conf) in detections:
                csv_writer.writerow([frame_id, label, yolo_conf, effnet_conf])

        # Calculate FPS safely
        curr_time = time.time()
        elapsed = curr_time - prev_time
        fps_val = 1.0 / elapsed if elapsed > 0 else 0.0
        prev_time = curr_time

        # Draw FPS
        cv2.putText(frame, f"FPS: {fps_val:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Draw class counts (per frame only)
        y_offset = 70
        for cls, count in class_counts.items():
            cv2.putText(frame, f"{cls}: {count}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[cls], 2)
            y_offset += 30

        # Show output
        cv2.imshow("YOLO + EfficientNet Pipeline", frame)

        # Save output if enabled
        if save_output and writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    if csv_file:
        csv_file.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_pipeline(0, save_output=True, output_path="advanced_output.mp4", log_csv=True)

