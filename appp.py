import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
from datetime import datetime
from PIL import Image

# ================================
# CONFIG
# ================================
st.set_page_config(
    page_title="♻️ Household Waste Wizard",
    page_icon="♻️",
    layout="wide"
)

# ⚠️ Match your dataset folder names exactly
CLASS_LABELS = ["hazardous", "organic", "recycleable"]
CLASS_COLORS = {
    "hazardous": (255, 0, 0),   # Red
    "organic": (0, 255, 0),     # Green
    "recycleable": (0, 0, 255)  # Blue
}
COLOR_NAMES = {"hazardous": "red", "organic": "green", "recycleable": "blue"}

# ================================
# LOAD MODEL
# ================================
@st.cache_resource
def load_model(provider="CPUExecutionProvider"):
    onnx_model_path = "efficientnet/effnet_b2.onnx"
    session = ort.InferenceSession(
        onnx_model_path,
        providers=[provider]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    expected_h, expected_w = session.get_inputs()[0].shape[2:4]
    effnet_size = expected_h if isinstance(expected_h, int) else 224
    return session, input_name, output_name, effnet_size

# ================================
# PREDICTION FUNCTION
# ================================
def predict_image(session, input_name, output_name, img, effnet_size):
    # Convert BGR → RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img_resized = cv2.resize(rgb, (effnet_size, effnet_size), interpolation=cv2.INTER_AREA)

    # Normalize [0,1]
    img_input = img_resized.astype(np.float32) / 255.0

    # EfficientNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, None, :]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, None, :]
    img_input = (img_input - mean) / std

    # HWC → CHW
    img_input = np.transpose(img_input, (2, 0, 1)).astype(np.float32)
    img_input = np.expand_dims(img_input, axis=0)

    preds = session.run([output_name], {input_name: img_input})[0]

    # Softmax
    exp_preds = np.exp(preds - np.max(preds, axis=1, keepdims=True))
    softmax_preds = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)

    pred_class = int(np.argmax(softmax_preds, axis=1)[0])
    confidence = float(np.max(softmax_preds, axis=1)[0])

    return softmax_preds[0], pred_class, confidence

# ================================
# DRAW BOX
# ================================
def draw_bounding_box(frame, label, confidence, color):
    h, w = frame.shape[:2]
    inset = 10
    cv2.rectangle(frame, (inset, inset), (w - inset, h - inset), color, 3)

    text = f"{label.capitalize()} ({confidence*100:.1f}%)"
    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    return frame

# ================================
# SIDEBAR
# ================================
st.sidebar.title("⚙️ Controls")
provider = st.sidebar.radio("Execution Provider:", ["CPUExecutionProvider", "CUDAExecutionProvider"])
mode = st.sidebar.radio("Choose Mode:", ["Upload Image", "Capture Image", "Real-Time Webcam"])
fps = st.sidebar.slider("FPS (refresh rate)", 1, 30, 10)
max_history = st.sidebar.slider("Prediction History Length", 5, 50, 10)
save_log = st.sidebar.checkbox("💾 Save Predictions to CSV")

camera_source = st.sidebar.radio("Camera Source:", ["Laptop Webcam", "Mobile (IP Webcam)"])
ip_url = st.sidebar.text_input("📱 Mobile Camera URL (if selected):", "http://192.168.0.118:8080/video")

with st.spinner("Loading ONNX model..."):
    session, input_name, output_name, effnet_size = load_model(provider)

# ================================
# MAIN APP
# ================================
st.title("♻️ Smart Waste Classification Dashboard")
st.write("Classify waste into **Hazardous**, **Organic**, or **Recycleable**.")

# ================================
# 1. UPLOAD IMAGE
# ================================
if mode == "Upload Image":
    uploaded_file = st.file_uploader("📂 Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img = np.array(image)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        with st.spinner("Analyzing..."):
            preds, pred_class, confidence = predict_image(session, input_name, output_name, img_bgr, effnet_size)

        label = CLASS_LABELS[pred_class]
        color = CLASS_COLORS[label]
        img_box = draw_bounding_box(img_bgr, label, confidence, color)

        st.image(cv2.cvtColor(img_box, cv2.COLOR_BGR2RGB), caption="Analyzed Image", use_column_width=True)
        st.markdown(f"### 🏷 Prediction: **{label.capitalize()}** ({confidence*100:.2f}%)")

        fig, ax = plt.subplots()
        bar_colors = [COLOR_NAMES[lbl] for lbl in CLASS_LABELS]
        ax.bar(CLASS_LABELS, preds, color=bar_colors)
        ax.set_ylabel("Confidence")
        ax.set_ylim([0, 1])
        st.pyplot(fig)

# ================================
# 2. CAPTURE IMAGE
# ================================
elif mode == "Capture Image":
    camera_image = st.camera_input("📸 Capture waste image")
    if camera_image:
        image = Image.open(camera_image).convert("RGB")
        img = np.array(image)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        with st.spinner("Analyzing..."):
            preds, pred_class, confidence = predict_image(session, input_name, output_name, img_bgr, effnet_size)

        label = CLASS_LABELS[pred_class]
        color = CLASS_COLORS[label]
        img_box = draw_bounding_box(img_bgr, label, confidence, color)

        st.image(cv2.cvtColor(img_box, cv2.COLOR_BGR2RGB), caption="Captured Image", use_column_width=True)
        st.markdown(f"### 🏷 Prediction: **{label.capitalize()}** ({confidence*100:.2f}%)")

        fig, ax = plt.subplots()
        bar_colors = [COLOR_NAMES[lbl] for lbl in CLASS_LABELS]
        ax.bar(CLASS_LABELS, preds, color=bar_colors)
        ax.set_ylabel("Confidence")
        ax.set_ylim([0, 1])
        st.pyplot(fig)

# ================================
# 3. REAL-TIME WEBCAM
# ================================
elif mode == "Real-Time Webcam":
    st.write("🎥 Real-time classification from webcam")

    col1, col2 = st.columns([2, 1])
    FRAME_WINDOW = col1.empty()
    chart_placeholder = col2.empty()
    history_placeholder = st.empty()

    pred_history = deque(maxlen=max_history)
    log_data = []

    run = st.checkbox("▶️ Start Webcam")

    if camera_source == "Laptop Webcam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(ip_url)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to access camera. Check your webcam or IP camera link.")
            break

        preds, pred_class, confidence = predict_image(session, input_name, output_name, frame, effnet_size)
        label = CLASS_LABELS[pred_class]
        color = CLASS_COLORS[label]

        frame = draw_bounding_box(frame, label, confidence, color)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Camera Feed")

        fig, ax = plt.subplots()
        bar_colors = [COLOR_NAMES[lbl] for lbl in CLASS_LABELS]
        ax.bar(CLASS_LABELS, preds, color=bar_colors)
        ax.set_ylim([0, 1])
        ax.set_ylabel("Confidence")
        ax.set_title("Prediction Probabilities")
        chart_placeholder.pyplot(fig)

        pred_history.appendleft((label, f"{confidence*100:.1f}%"))
        history_placeholder.table({
            "Label": [h[0] for h in pred_history],
            "Confidence": [h[1] for h in pred_history]
        })

        if save_log:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_data.append({"time": timestamp, "label": label, "confidence": confidence})

        time.sleep(1.0 / fps)

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
