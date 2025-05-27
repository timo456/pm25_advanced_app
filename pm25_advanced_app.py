# ✅ Unified PM2.5 Detection App with Full History + ML Logic
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime, timedelta
from fpdf import FPDF

st.set_page_config(page_title="PM2.5 多功能預測工具", layout="wide")
st.title("🌫️ 空氣品質辨識系統（完整進階版）")

@st.cache_data
def load_model():
    return joblib.load("pm25_model.pkl")

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

def predict_pm25(features, model):
    proba = model.predict_proba([features])[0][1]
    if features[0] < 0.5 and features[1] < 10 and features[4] < 100:
        return "天氣不明", "無法判斷"
    elif proba > 0.7:
        return "模糊影像", "超標"
    elif proba < 0.3:
        return "清晰影像", "未超標"
    else:
        return "邊界模糊", "無法判斷"

def save_to_db(data):
    conn = sqlite3.connect("pm25_history.db")
    df = pd.DataFrame(data)
    df["時間"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.to_sql("history", conn, if_exists="append", index=False)
    conn.close()

def query_history(start, end):
    conn = sqlite3.connect("pm25_history.db")
    query = f"SELECT * FROM history WHERE 時間 BETWEEN '{start}' AND '{end}'"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def show_history_chart(df):
    df["時間"] = pd.to_datetime(df["時間"])
    daily = df.groupby(df["時間"].dt.date)["PM2.5狀態"].value_counts().unstack().fillna(0)
    st.bar_chart(daily)

def generate_pdf(data, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="PM2.5 Report", ln=True, align="C")
    pdf.ln(10)

    headers = ["Filename", "BlueSky(%)", "Laplacian", "PM2.5"]
    for h in headers:
        pdf.cell(45, 10, h, border=1, align="C")
    pdf.ln()

    for row in data:
        pm25_status = row["PM2.5狀態"]
        if pm25_status == "未超標":
            pdf.set_text_color(0, 128, 0)
            pm25_status_en = "Normal"
        elif pm25_status == "超標":
            pdf.set_text_color(255, 0, 0)
            pm25_status_en = "High"
        else:
            pdf.set_text_color(0, 0, 255)
            pm25_status_en = "Unknown"

        pdf.cell(45, 10, row["檔名"], border=1)
        pdf.cell(45, 10, str(round(row["藍天比例(%)"], 2)), border=1)
        pdf.cell(45, 10, str(round(row["Laplacian平均值"], 2)), border=1)
        pdf.cell(45, 10, pm25_status_en, border=1)
        pdf.set_text_color(0, 0, 0)
        pdf.ln()
    pdf.output(filename)

# 載入模型
model = load_model()

# UI 分頁
tab1, tab2, tab3 = st.tabs(["📷 即時攝影預測", "📂 圖片批次分析", "📅 歷史查詢與圖表"])

# 📷 即時預測
with tab1:
    st.subheader("📷 使用攝影機即時預測")
    picture = st.camera_input("請啟用攝影機")
    if picture:
        file_bytes = np.frombuffer(picture.getvalue(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        blue, lap, lap_std, sobel, brightness = analyze_image(image)
        result, pm25 = predict_pm25([blue, lap, lap_std, sobel, brightness], model)

        st.image(image, channels="BGR", caption="即時影像", use_column_width=True)
        st.markdown(f"- 藍天比例：{blue:.2f}%")
        st.markdown(f"- Laplacian 清晰度：{lap:.2f}")
        st.success("✅ PM2.5 狀態：" + pm25)

        save_to_db([{
            "檔名": "camera.jpg",
            "藍天比例(%)": round(blue, 2),
            "Laplacian平均值": round(lap, 2),
            "PM2.5狀態": pm25
        }])

# 📁 圖片批次分析
with tab2:
    st.subheader("📂 上傳圖片進行預測")
    uploaded_files = st.file_uploader("選擇圖片（可多選）", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        results = []
        for file in uploaded_files:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            blue, lap, lap_std, sobel, brightness = analyze_image(image)
            result, pm25 = predict_pm25([blue, lap, lap_std, sobel, brightness], model)
            results.append({
                "檔名": file.name,
                "藍天比例(%)": round(blue, 2),
                "Laplacian平均值": round(lap, 2),
                "PM2.5狀態": pm25
            })

        df_result = pd.DataFrame(results)
        st.dataframe(df_result)
        save_to_db(results)

        csv = df_result.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 下載 CSV", csv, file_name="PM2.5預測結果.csv", mime="text/csv")

        pdf_path = "PM2.5_報告.pdf"
        generate_pdf(results, pdf_path)
        with open(pdf_path, "rb") as f:
            st.download_button("🧾 下載 PDF 報告", f.read(), file_name=pdf_path, mime="application/pdf")

# 📅 歷史查詢與圖表
with tab3:
    st.subheader("📅 歷史查詢與趨勢")
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("開始日期", value=datetime.now() - timedelta(days=7))
    with col2:
        end = st.date_input("結束日期", value=datetime.now())

    if st.button("🔍 查詢"):
        df = query_history(str(start), str(end + timedelta(days=1)))
        if df.empty:
            st.warning("找不到符合的資料")
        else:
            st.dataframe(df)
            st.subheader("📈 PM2.5 狀態趨勢圖")
            show_history_chart(df)
