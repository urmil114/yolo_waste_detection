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

# Non-waste classes to reject
NON_WASTE_CLASSES = {
    "person", "dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "chair", "sofa", "bed", "dining table", "tv", "laptop"
}

# ================================
# LOAD EFFICIENTNET
# ================================
@st.cache_resource
def load_effnet(provider="CPUExecutionProvider"):
    onnx_model_path = "efficientnet/effnet_b2.onnx"
    session = ort.InferenceSession(onnx_model_path, providers=[provider])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    effnet_size = 224
    return session, input_name, output_name, effnet_size

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
# VISUALS
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
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Class", sort=CLASS_LABELS),
            y=alt.Y("Confidence", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Class", scale=alt.Scale(domain=CLASS_LABELS,
                                                     range=[CLASS_COLORS[c] for c in CLASS_LABELS]))
        )
        .properties(width=300, height=300)
    )
    return chart

# ================================
# YOLO + EFFNET PIPELINE (FIXED with brightness check)
# ================================
def handle_prediction(img_bgr, session, input_name, output_name, effnet_size):
    # ---- Check brightness (black screen detector) ----
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    if brightness < 30:  # threshold for dark/black frame
        return None, "no-detect", None, img_bgr

    # YOLO detection
    results = yolo_model.predict(img_bgr, verbose=False)

    if len(results[0].boxes) > 0:
        detected_classes = set([results[0].names[int(cls)] for cls in results[0].boxes.cls])
        if all(obj in NON_WASTE_CLASSES for obj in detected_classes):
            return None, "non-waste", None, img_bgr

    # EfficientNet classification
    preds, pred_class, confidence = predict_image(session, input_name, output_name, img_bgr, effnet_size)
    label = CLASS_LABELS[pred_class]
    bgr_color = tuple(int(CLASS_COLORS[label].lstrip("#")[i:i+2], 16) for i in (4, 2, 0))

    if confidence < CONFIDENCE_THRESHOLD:
        return None, "low-confidence", None, img_bgr

    img_box = draw_bounding_box(img_bgr, label, confidence, bgr_color)
    return preds, label, confidence, img_box

# ================================
# SIDEBAR
# ================================
st.sidebar.title("⚙️ Controls")
provider = st.sidebar.radio("Execution Provider:", ["CPUExecutionProvider", "CUDAExecutionProvider"])
mode = st.sidebar.radio("Choose Mode:", ["Upload Image", "Capture Image", "Real-Time Webcam"])
max_history = st.sidebar.slider("Prediction History Length", 5, 50, 10)
save_log = st.sidebar.checkbox("💾 Save Predictions to CSV")

camera_source = st.sidebar.radio("Camera Source:", ["Laptop Webcam", "Mobile (IP Webcam)"])
ip_url = st.sidebar.text_input("📱 Mobile Camera URL (if selected):", "http://192.168.0.118:8080/video")

with st.spinner("Loading EfficientNet model..."):
    session, input_name, output_name, effnet_size = load_effnet(provider)

# ================================
# 1. UPLOAD IMAGE
# ================================
if mode == "Upload Image":
    uploaded_file = st.file_uploader("📂 Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        preds, label, confidence, img_box = handle_prediction(img_bgr, session, input_name, output_name, effnet_size)

        if label == "no-detect":
            st.warning("⚠️ Waste not detected (black/empty frame).")
            st.image(image, caption="Uploaded Image", use_container_width=True)
        elif label == "non-waste":
            st.warning("⚠️ Non-waste object detected.")
            st.image(image, caption="Uploaded Image", use_container_width=True)
        elif label == "low-confidence":
            st.warning("⚠️ Low confidence. Try another image.")
            st.image(image, caption="Uploaded Image", use_container_width=True)
        elif preds is not None:
            st.image(cv2.cvtColor(img_box, cv2.COLOR_BGR2RGB), caption="Analyzed Image", use_container_width=True)
            st.altair_chart(plot_bar_chart(preds), use_container_width=True)

# ================================
# 2. CAPTURE IMAGE
# ================================
elif mode == "Capture Image":
    camera_image = st.camera_input("📸 Capture waste image")
    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        preds, label, confidence, img_box = handle_prediction(img_bgr, session, input_name, output_name, effnet_size)

        if label == "no-detect":
            st.warning("⚠️ Waste not detected (black/empty frame).")
            st.image(image, caption="Captured Image", use_container_width=True)
        elif label == "non-waste":
            st.warning("⚠️ Non-waste object detected.")
            st.image(image, caption="Captured Image", use_container_width=True)
        elif label == "low-confidence":
            st.warning("⚠️ Low confidence.")
            st.image(image, caption="Captured Image", use_container_width=True)
        elif preds is not None:
            st.image(cv2.cvtColor(img_box, cv2.COLOR_BGR2RGB), caption="Analyzed Image", use_container_width=True)
            st.altair_chart(plot_bar_chart(preds), use_container_width=True)

# ================================
# 3. REAL-TIME WEBCAM
# ================================
elif mode == "Real-Time Webcam":
    st.title("🎥 Real-Time Waste Detection & Classification")

    col1, col2 = st.columns([2, 1])
    FRAME_WINDOW = col1.empty()
    chart_placeholder = col2.empty()
    history_placeholder = st.empty()

    pred_history = deque(maxlen=max_history)
    log_data = []

    run = st.checkbox("▶️ Start Webcam")

    if camera_source == "Laptop Webcam":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(ip_url)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while run:
        ret, frame = cap.read()
        if not ret or frame is None:
            st.warning("⚠️ No frame received from camera.")
            continue

        preds, label, confidence, frame_box = handle_prediction(frame, session, input_name, output_name, effnet_size)

        if label == "no-detect":
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                               caption="⚠️ Waste not detected (black/empty frame)", channels="RGB")
            continue
        elif label == "non-waste":
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                               caption="⚠️ Non-waste object detected", channels="RGB")
            continue
        elif label == "low-confidence":
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                               caption="⚠️ Low confidence, try again", channels="RGB")
            continue
        elif preds is None:
            continue

        FRAME_WINDOW.image(cv2.cvtColor(frame_box, cv2.COLOR_BGR2RGB), caption="Camera Feed", channels="RGB")
        chart_placeholder.altair_chart(plot_bar_chart(preds), use_container_width=True)

        pred_history.appendleft((label, f"{confidence*100:.1f}%"))
        history_df = pd.DataFrame(pred_history, columns=["Label", "Confidence"])
        history_placeholder.table(history_df)

        if save_log:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_data.append({"time": timestamp, "label": label, "confidence": confidence})

    cap.release()

    if save_log and log_data:
        df = pd.DataFrame(log_data)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Predictions Log",
            data=csv,
            file_name="waste_predictions_log.csv",
            mime="text/csv"
        )
