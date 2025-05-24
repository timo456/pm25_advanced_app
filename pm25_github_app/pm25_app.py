import streamlit as st
import pandas as pd
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="PM2.5 預測工具", layout="wide")
st.title("🌫️ PM2.5 空氣品質預測 Demo")

@st.cache_data
def load_model():
    return joblib.load("air_quality_model.pkl")

def analyze_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (100, 150, 0), (140, 255, 255))
    blue_ratio = cv2.countNonZero(mask) / (image.shape[0] * image.shape[1])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_mean = np.mean(np.abs(lap))
    return blue_ratio * 100, lap_mean

st.subheader("📂 上傳圖片預測 PM2.5 狀態")

uploaded_file = st.file_uploader("請上傳戶外圖片", type=["jpg", "jpeg", "png"])
if uploaded_file:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    blue, lap = analyze_image(image)
    df = pd.DataFrame([[blue, lap]], columns=["藍天比例(%)", "Laplacian平均值"])
    model = load_model()
    pred = model.predict(df)[0]
    pm25 = "未超標" if pred == 0 else "超標"

    st.image(image, channels="BGR", caption="上傳影像", use_column_width=True)
    st.markdown(f"🔵 藍天比例：{blue:.2f}%")
    st.markdown(f"📈 Laplacian 清晰度：{lap:.2f}")
    if pred == 0:
        st.success("✅ PM2.5 預測結果：未超標")
    else:
        st.error("❌ PM2.5 預測結果：超標")
