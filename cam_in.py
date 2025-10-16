import cv2
import numpy as np
import time
import datetime
import platform
import sys
import csv
from collections import defaultdict
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
# 2. Load EfficientNet ONNX
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

# Class labels (⚠️ adjust order if mismatch)
class_labels = ["hazardous", "organic", "recyclable"]

# Colors
colors = {
    "hazardous": (0, 0, 255),   # Red
    "organic": (0, 255, 0),     # Green
    "recyclable": (255, 165, 0) # Orange
}

# ================================
# 3. Preprocess Function
# ================================
def preprocess(image, size=effnet_size):
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
# 4. Process Frame (EfficientNet only)
# ================================
def process_frame(frame, class_counts, conf_thresh=50.0):
    detections = []

    # Whole frame classification
    input_tensor = preprocess(frame)
    pred = session.run([output_name], {input_name: input_tensor})[0]
    class_id = int(np.argmax(pred))

    # Convert confidence to %
    effnet_conf = float(np.max(pred)) * 100
    effnet_conf = min(max(effnet_conf, 0), 100)

    label = class_labels[class_id]

    if effnet_conf >= conf_thresh:
        class_counts[label] += 1
        detections.append((label, effnet_conf))
        color = colors[label]

        # Bounding box = full frame (since no YOLO here)
        h, w, _ = frame.shape
        cv2.rectangle(frame, (5, 5), (w - 5, h - 5), color, 3)

        # Draw classification result at top
        text = f"{label} ({effnet_conf:.1f}%)"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (10, 10), (10 + tw + 10, 10 + th + 20), color, -1)
        cv2.putText(frame, text, (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return frame, detections

# ================================
# 5. Main Loop
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
        csv_writer.writerow(["Frame", "Label", "EffNet_Conf(%)"])

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
        for label, effnet_conf in detections:
            if csv_writer:
                csv_writer.writerow([frame_id, label, f"{effnet_conf:.1f}%"])

            if label == "hazardous" and effnet_conf > 75:
                alert(1000, 400)
                print("⚠️ Hazardous waste detected!")

            elif label == "organic" and effnet_conf > 75:
                alert(600, 200)
                print("🌱 Organic waste detected.")

            elif label == "recyclable" and effnet_conf > 75:
                alert(800, 200)
                print("♻️ Recyclable waste detected.")

        # FPS calculation
        curr_time = time.time()
        fps_val = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_smooth = (fps_smooth * 0.9) + (fps_val * 0.1)

        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Class counts
        y_offset = 110
        for cls, count in class_counts.items():
            cv2.putText(frame, f"{cls}: {count}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[cls], 2)
            y_offset += 30

        cv2.imshow("EfficientNet-B2 Inference", frame)

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
# 6. Run
# ================================
if __name__ == "__main__":
    run(0, save_output=True, output_path="advanced_output.mp4", log_csv=True)
