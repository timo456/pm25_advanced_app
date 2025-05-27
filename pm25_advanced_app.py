# ✅ Unified PM2.5 Detection App with ML-Based Logic (改進版)
import streamlit as st
st.set_page_config(page_title="PM2.5 預測系統（機器學習版）", layout="wide")
import pandas as pd
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
import sqlite3
from datetime import datetime, timedelta
import matplotlib.font_manager as fm

# 設定 matplotlib 使用中文字型
font_path = "fonts/NotoSerifTC-Regular.ttf"
my_font = fm.FontProperties(fname=font_path)
plt.rcParams["axes.unicode_minus"] = False
st.text(f"✅ 目前載入的字型為: {my_font.get_name()}")

st.title("🌫️ PM2.5 空氣品質多功能預測系統（機器學習）")

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

model = joblib.load("pm25_model.pkl")

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
