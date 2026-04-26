# 🪖 Bike Helmet Detection System

A real-time bike helmet detection system using YOLOv8 object detection model.

## 🎯 About
This application detects whether bike riders are wearing helmets or not using live camera or uploaded images. It generates a safety compliance report showing helmet wearing percentage.

## ✨ Features
- 📷 **Image Upload** — Upload any bike rider image for detection
- 🎥 **Live Camera** — Real-time detection using webcam
- 📊 **Safety Report** — Shows compliance percentage
- ⚠️ **Alert System** — Warning when helmet not detected

## 🛠️ Tech Stack
- **Model:** YOLOv8 (Ultralytics)
- **Dataset:** Kaggle - Rider With Helmet Without Helmet
- **Web App:** Streamlit
- **Image Processing:** OpenCV, Pillow
- **Training:** Google Colab (Tesla T4 GPU)
- **Deployment:** Streamlit Cloud

## 📊 Model Performance
- **Accuracy:** 74% mAP50
- **Training:** 20 epochs
- **Dataset:** 764 bike rider images
- **Classes:** with helmet, without helmet, rider, number plate

## 🚀 Live Demo
👉 [Click here to try the app](https://bike-helmet-detection-ag5zdbcqyqgvnogtukasni.streamlit.app)

## ⚙️ How to Run Locally
```bash
git clone https://github.com/Pratyusha-31/bike-helmet-detection.git
cd bike-helmet-detection
pip install -r requirements.txt
streamlit run app.py
```
