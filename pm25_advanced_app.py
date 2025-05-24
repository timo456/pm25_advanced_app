import streamlit as st
import pandas as pd
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime, timedelta

st.set_page_config(page_title="PM2.5 多功能預測工具", layout="wide")
st.title("🌫️ 空氣品質辨識系統（進階版）")

@st.cache_data
def load_model():
    return joblib.load("air_quality_model_cloud_ready.pkl")

def analyze_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (100, 150, 0), (140, 255, 255))
    blue_ratio = cv2.countNonZero(mask) / (image.shape[0] * image.shape[1])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_mean = np.mean(np.abs(lap))
    return blue_ratio * 100, lap_mean

def save_to_db(data):
    conn = sqlite3.connect("pm25_history.db")
    df = pd.DataFrame(data)
    df["時間"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.to_sql("history", conn, if_exists="append", index=False)
    conn.close()

def query_history(start, end):
    conn = sqlite3.connect("pm25_history.db")
    query = f"""
        SELECT * FROM history
        WHERE 時間 BETWEEN '{start}' AND '{end}'
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def show_history_chart(df):
    df["時間"] = pd.to_datetime(df["時間"])
    daily = df.groupby(df["時間"].dt.date)["PM2.5狀態"].value_counts().unstack().fillna(0)
    st.bar_chart(daily)

# 模型載入
model = load_model()

tab1, tab2, tab3 = st.tabs(["📷 即時攝影預測", "📂 圖片批次分析", "📅 歷史查詢與圖表"])

# Tab 1: Camera
with tab1:
    st.subheader("📷 即時攝影機拍照")
    picture = st.camera_input("請使用攝影機拍照")
    if picture:
        file_bytes = np.frombuffer(picture.getvalue(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        blue, lap = analyze_image(image)
        df = pd.DataFrame([[blue, lap]], columns=["藍天比例(%)", "Laplacian平均值"])
        pred = model.predict(df)[0]
        pm25 = "未超標" if pred == 0 else "超標"

        st.image(image, channels="BGR", caption="即時影像", use_column_width=True)
        st.markdown(f"- 藍天比例：{blue:.2f}%")
        st.markdown(f"- Laplacian 清晰度：{lap:.2f}")
        st.success("✅ 預測結果：PM2.5 未超標" if pred == 0 else "❌ 預測結果：PM2.5 超標")

        save_to_db([{
            "檔名": "camera.jpg",
            "藍天比例(%)": round(blue, 2),
            "Laplacian平均值": round(lap, 2),
            "PM2.5狀態": pm25
        }])

# Tab 2: File upload
with tab2:
    st.subheader("📁 批次圖片預測")
    uploaded_files = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        results = []
        for file in uploaded_files:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            blue, lap = analyze_image(image)
            df = pd.DataFrame([[blue, lap]], columns=["藍天比例(%)", "Laplacian平均值"])
            pred = model.predict(df)[0]
            pm25 = "未超標" if pred == 0 else "超標"

            results.append({
                "檔名": file.name,
                "藍天比例(%)": round(blue, 2),
                "Laplacian平均值": round(lap, 2),
                "PM2.5狀態": pm25
            })

        result_df = pd.DataFrame(results)
        st.dataframe(result_df)
        save_to_db(results)

        csv = result_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 下載預測結果", csv, file_name="PM2.5預測結果.csv", mime="text/csv")

# Tab 3: History
with tab3:
    st.subheader("📅 歷史查詢")
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
