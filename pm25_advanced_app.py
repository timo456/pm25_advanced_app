# ✅ Unified PM2.5 Detection App with ML-Based Logic (完整)
import streamlit as st
st.set_page_config(page_title="PM2.5 預測系統（機器學習版）", layout="wide")
import pandas as pd
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime, timedelta
import matplotlib.font_manager as fm

# 字型設定
font_path = "fonts/NotoSerifTC-Regular.ttf"
my_font = fm.FontProperties(fname=font_path)
plt.rcParams["axes.unicode_minus"] = False
st.text(f"✅ 目前載入的字型為: {my_font.get_name()}")
st.title("🌫️ PM2.5 空氣品質多功能預測系統（機器學習）")

# 模型載入
model = joblib.load("pm25_model.pkl")

# 特徵分析函數
def analyze_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (100, 150, 0), (140, 255, 255))
    blue_ratio = cv2.countNonZero(mask) / (image.shape[0] * image.shape[1])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_mean = np.mean(np.abs(lap))
    lap_std = np.std(lap)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_mean = np.mean(sobel_mag)
    brightness = np.mean(gray)
    return blue_ratio * 100, lap_mean, lap_std, sobel_mean, brightness

# 預測函數
def ml_model_prediction(blue, lap, lap_std, sobel, brightness):
    features = [[blue, lap, lap_std, sobel, brightness]]
    proba = model.predict_proba(features)[0][1]
    if blue < 0.5 and lap < 10 and brightness < 100:
        return "天氣不明", "無法判斷"
    elif proba > 0.7:
        return "模糊影像", "超標"
    elif proba < 0.3:
        return "清晰影像", "未超標"
    else:
        return "邊界模糊", "無法判斷"

# Tabs
tab1, tab2 = st.tabs(["📁 圖片分析", "📈 模型說明"])

# 圖片上傳分析
with tab1:
    st.subheader("📁 上傳圖片進行分析")
    uploaded_files = st.file_uploader("上傳圖片 (支援多張)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            blue, lap, lap_std, sobel, brightness = analyze_image(image)
            result, pm25 = ml_model_prediction(blue, lap, lap_std, sobel, brightness)

            st.image(image, channels="BGR", caption=file.name, use_column_width=True)
            st.markdown(f"- 藍天比例：{blue:.2f}%")
            st.markdown(f"- Laplacian 清晰度：{lap:.2f}")
            st.markdown(f"- Laplacian 標準差：{lap_std:.2f}")
            st.markdown(f"- Sobel 邊緣強度：{sobel:.2f}")
            st.markdown(f"- 平均亮度：{brightness:.2f}")
            st.info(f"🔍 預測判斷：{result}")
            st.success("✅ PM2.5 狀態：未超標" if pm25 == "未超標" else "❌ PM2.5 狀態：超標" if pm25 == "超標" else "⚠️ PM2.5 狀態：無法判斷")

# 模型說明
with tab2:
    st.subheader("🔍 模型使用說明")
    st.markdown("""
    - 本系統使用 **五大影像特徵** 進行機器學習判斷：
        1. 藍天比例
        2. Laplacian 平均值（清晰度）
        3. Laplacian 標準差（變異性）
        4. Sobel 邊緣強度
        5. 平均亮度

    - 模型使用 `XGBoost` 訓練，透過 `predict_proba()` 決定「模糊 / 清晰 / 無法判斷」
    - 如果藍天太少、畫面太暗，也可能被標記為「天氣不明」

    ✨ 歡迎上傳不同天氣與畫質圖片觀察結果！
    """)
