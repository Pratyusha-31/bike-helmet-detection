import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(
    page_title="Bike Helmet Detection",
    page_icon="🪖",
    layout="wide"
)

st.title("🪖 Bike Helmet Detection System")
st.markdown("**Real-time bike helmet detection using YOLOv8**")
st.divider()

@st.cache_resource
def load_model():
    return YOLO('bike_best.pt')

model = load_model()

classes = ['with helmet', 'without helmet', 'rider', 'number plate']

tab1, tab2 = st.tabs(["📷 Upload Image", "🎥 Live Camera"])

with tab1:
    st.subheader("Upload Image Detection")
    confidence = st.slider("Confidence Level", 0.1, 1.0, 0.5)

    uploaded_file = st.file_uploader(
        "Upload an image of a bike rider",
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)

        results = model.predict(image, conf=confidence)
        result_img = results[0].plot()
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        with col2:
            st.subheader("Detection Result")
            st.image(result_img, use_column_width=True)

        st.divider()
        st.subheader("📊 Safety Report")

        with_helmet = 0
        without_helmet = 0
        for box in results[0].boxes:
            cls = int(box.cls[0])
            cls_name = classes[cls]
            if cls_name == 'with helmet':
                with_helmet += 1
            elif cls_name == 'without helmet':
                without_helmet += 1

        col3, col4, col5 = st.columns(3)
        col3.metric("✅ With Helmet", with_helmet)
        col4.metric("❌ Without Helmet", without_helmet)
        total = with_helmet + without_helmet
        percent = (with_helmet/total*100) if total > 0 else 0
        col5.metric("📈 Compliance %", f"{percent:.1f}%")

        if without_helmet > 0:
            st.error(f"⚠️ {without_helmet} rider(s) NOT wearing helmet!")
        elif with_helmet > 0:
            st.success("✅ All riders wearing helmets. Safe!")
        else:
            st.warning("No riders detected in image.")

with tab2:
    st.subheader("🎥 Live Camera Detection")
    confidence_cam = st.slider("Confidence", 0.1, 1.0, 0.5, key="cam")

    run = st.checkbox("▶ Start Camera")
    FRAME_WINDOW = st.image([])
    status_box = st.empty()

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not found!")
            break

        results = model.predict(frame, conf=confidence_cam, verbose=False)
        result_frame = results[0].plot()
        result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(result_frame)

        with_helmet = 0
        without_helmet = 0
        for box in results[0].boxes:
            cls = int(box.cls[0])
            cls_name = classes[cls]
            if cls_name == 'with helmet':
                with_helmet += 1
            elif cls_name == 'without helmet':
                without_helmet += 1

        if without_helmet > 0:
            status_box.error(f"⚠️ {without_helmet} rider(s) NOT wearing helmet!")
        elif with_helmet > 0:
            status_box.success("✅ Helmet detected! You are safe!")
        else:
            status_box.info("👀 No rider detected...")

    cap.release()