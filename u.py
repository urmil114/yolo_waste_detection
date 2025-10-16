import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
import pandas as pd
from collections import deque
from datetime import datetime
import altair as alt
from PIL import Image
from ultralytics import YOLO

# ================================
# CONFIG
# ================================
st.set_page_config(
    page_title="♻️ Household Waste Wizard",
    page_icon="♻️",
    layout="wide"
)

CLASS_LABELS = ["hazardous", "organic", "recycleable"]
CLASS_COLORS = {
    "hazardous": "#FF0000",   # Red
    "organic": "#00FF00",     # Green
    "recycleable": "#0000FF"  # Blue
}
CONFIDENCE_THRESHOLD = 0.75
YOLO_MODEL_PATH = "yolov8s.pt"
yolo_model = YOLO(YOLO_MODEL_PATH)

NON_WASTE_CLASSES = {
    "person", "dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "chair", "sofa", "bed", "dining table", "tv", "laptop"
}

FORCE_RECYCLE_CLASSES = {
    "pen", "pencil", "marker", "stationary", "stationery",
    "crayon", "highlighter", "sketch", "bottle", "cup",
    "plastic", "toothbrush", "fork", "knife", "spoon"
}

# ================================
# LOAD MODELS
# ================================
@st.cache_resource
def load_effnet(provider="CPUExecutionProvider"):
    session = ort.InferenceSession(
        "efficientnet/effnet_b2.onnx",
        providers=[provider]
    )
    return session, session.get_inputs()[0].name, session.get_outputs()[0].name, 224


# ================================
# EFFICIENTNET PREDICTION
# ================================
def predict_image(session, input_name, output_name, img, effnet_size):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(rgb, (effnet_size, effnet_size))
    img_input = img_resized.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, None, :]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, None, :]
    img_input = (img_input - mean) / std
    img_input = np.transpose(img_input, (2, 0, 1))[None, :, :, :]

    preds = session.run([output_name], {input_name: img_input})[0]
    exp_preds = np.exp(preds - np.max(preds, axis=1, keepdims=True))
    softmax_preds = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)

    pred_class = int(np.argmax(softmax_preds, axis=1)[0])
    confidence = float(np.max(softmax_preds, axis=1)[0])
    return softmax_preds[0], pred_class, confidence


# ================================
# VISUAL HELPERS
# ================================
def draw_bounding_box(frame, label, confidence, bgr_color):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (10, 10), (w - 10, h - 10), bgr_color, 3)
    text = f"{label.capitalize()} ({confidence*100:.1f}%)"
    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, bgr_color, 2, cv2.LINE_AA)
    return frame


def plot_bar_chart(preds):
    df = pd.DataFrame({
        "Class": CLASS_LABELS,
        "Confidence": preds,
        "Color": [CLASS_COLORS[c] for c in CLASS_LABELS]
    })
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Class", sort=CLASS_LABELS),
            y=alt.Y("Confidence", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Class", scale=alt.Scale(
                domain=CLASS_LABELS,
                range=[CLASS_COLORS[c] for c in CLASS_LABELS]
            ))
        )
        .properties(width=300, height=300)
    )


# ================================
# YOLO + EFFNET PIPELINE
# ================================
def handle_prediction(img_bgr, session, input_name, output_name, effnet_size):
    if np.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)) < 30:
        return None, "no-detect", None, img_bgr

    results = yolo_model.predict(img_bgr, verbose=False)
    detected_classes = set(results[0].names[int(cls)]
                           for cls in results[0].boxes.cls) if len(results[0].boxes) else set()

    if detected_classes and all(obj in NON_WASTE_CLASSES for obj in detected_classes):
        return None, "non-waste", None, img_bgr

    if any(obj.lower() in FORCE_RECYCLE_CLASSES for obj in detected_classes):
        return [0.0, 0.0, 1.0], "recycleable", 1.0, draw_bounding_box(img_bgr, "recycleable", 1.0, (255, 0, 0))

    preds, pred_class, confidence = predict_image(
        session, input_name, output_name, img_bgr, effnet_size)
    label = CLASS_LABELS[pred_class]

    if label == "hazardous" and confidence >= 0.90 and (not detected_classes or all(obj not in NON_WASTE_CLASSES for obj in detected_classes)):
        preds, label, confidence = [0.0, 0.0, 1.0], "recycleable", 1.0

    if confidence < CONFIDENCE_THRESHOLD:
        return None, "low-confidence", None, img_bgr

    bgr_color = tuple(int(CLASS_COLORS[label].lstrip("#")[i:i+2], 16)
                      for i in (4, 2, 0))
    return preds, label, confidence, draw_bounding_box(img_bgr, label, confidence, bgr_color)


# ================================
# SIDEBAR CONTROLS
# ================================
st.sidebar.title("⚙️ Controls")
provider = st.sidebar.radio(
    "Execution Provider:", ["CPUExecutionProvider", "CUDAExecutionProvider"])
mode = st.sidebar.radio(
    "Choose Mode:", ["Upload Image", "Capture Image", "Real-Time Webcam"])
max_history = st.sidebar.slider("Prediction History Length", 5, 50, 10)
save_log = st.sidebar.checkbox("💾 Save Predictions to CSV")
camera_source = st.sidebar.radio(
    "Camera Source:", ["Laptop Webcam", "Mobile (IP Webcam)"])
ip_url = st.sidebar.text_input(
    "📱 Mobile Camera URL (if selected):", "http://192.168.0.118:8080/video")

with st.spinner("Loading EfficientNet model..."):
    session, input_name, output_name, effnet_size = load_effnet(provider)


# ================================
# PREDICTION HANDLER
# ================================
def process_and_display(img_bgr, source_caption="Image"):
    preds, label, confidence, img_box = handle_prediction(
        img_bgr, session, input_name, output_name, effnet_size)

    if label in ["no-detect", "non-waste", "low-confidence"]:
        st.warning(f"⚠️ {label.replace('-', ' ').capitalize()}.")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                 caption=source_caption, use_container_width=True)
        return None
    elif preds is not None:
        st.image(cv2.cvtColor(img_box, cv2.COLOR_BGR2RGB),
                 caption="Analyzed " + source_caption, use_container_width=True)
        st.altair_chart(plot_bar_chart(preds), use_container_width=True)
        return label, confidence
    return None


# ================================
# 1. UPLOAD IMAGE
# ================================
if mode == "Upload Image":
    uploaded_file = st.file_uploader(
        "📂 Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        process_and_display(img_bgr, "Uploaded Image")

# ================================
# 2. CAPTURE IMAGE
# ================================
elif mode == "Capture Image":
    camera_image = st.camera_input("📸 Capture waste image")
    if camera_image:
        image = Image.open(camera_image).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        process_and_display(img_bgr, "Captured Image")

# ================================
# 3. REAL-TIME WEBCAM
# ================================
elif mode == "Real-Time Webcam":
    st.title("🎥 Real-Time Waste Detection & Classification")
    col1, col2 = st.columns([2, 1])
    FRAME_WINDOW, chart_placeholder = col1.empty(), col2.empty()
    history_placeholder, pred_history, log_data = st.empty(), deque(
        maxlen=max_history), []

    run = st.checkbox("▶️ Start Webcam")

    # ✅ Open webcam only once
    if run:
        cap = cv2.VideoCapture(0 if camera_source == "Laptop Webcam" else ip_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ No frame received from camera.")
                continue

            result = process_and_display(frame, "Camera Feed")
            if result:
                label, confidence = result
                pred_history.appendleft((label, f"{confidence*100:.1f}%"))
                history_placeholder.table(pd.DataFrame(
                    pred_history, columns=["Label", "Confidence"]))
                if save_log:
                    log_data.append(
                        {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         "label": label, "confidence": confidence})

        cap.release()

    # Save log after session ends
    if save_log and log_data:
        df = pd.DataFrame(log_data)
        st.download_button(
            "⬇️ Download Predictions Log",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="waste_predictions_log.csv",
            mime="text/csv"
        )
