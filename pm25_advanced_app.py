# ✅ Unified PM2.5 Detection App with Rule-Based Logic
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
import sqlite3
from datetime import datetime, timedelta

plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="PM2.5 預測系統（整合版）", layout="wide")
st.title("🌫️ PM2.5 空氣品質多功能預測系統")

def analyze_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (100, 150, 0), (140, 255, 255))
    blue_ratio = cv2.countNonZero(mask) / (image.shape[0] * image.shape[1])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_mean = np.mean(np.abs(lap))
    brightness = np.mean(gray)
    return blue_ratio * 100, lap_mean, brightness

def rule_based_prediction(blue, lap, brightness):
    if blue < 0.5 and lap < 10 and brightness < 100:
        return "天氣不明", "無法判斷"
    elif lap < 10:
        return "模糊影像", "超標"
    else:
        return "清晰影像", "未超標"

def save_to_db(data):
    conn = sqlite3.connect("pm25_history.db")
    df = pd.DataFrame(data)
    df["時間"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "圖片縮圖" in df.columns:
        df.drop(columns=["圖片縮圖"], inplace=True)
    df.to_sql("history", conn, if_exists="append", index=False)
    conn.close()

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
        try:
            filename_ascii = row["檔名"].encode("latin-1").decode("latin-1")
        except UnicodeEncodeError:
            filename_ascii = "Non-ASCII"

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

        pdf.cell(45, 10, filename_ascii, border=1)
        pdf.cell(45, 10, str(row["藍天比例(%)"]), border=1)
        pdf.cell(45, 10, str(row["Laplacian平均值"]), border=1)
        pdf.cell(45, 10, pm25_status_en, border=1)
        pdf.set_text_color(0, 0, 0)
        pdf.ln()
    pdf.output(filename)

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

# 刪除舊的資料表
conn = sqlite3.connect("pm25_history.db")
conn.execute("DROP TABLE IF EXISTS history")
conn.close()

tab1, tab2, tab3 = st.tabs(["📷 即時預測", "📁 圖片分析", "📅 歷史查詢"])

with tab1:
    st.subheader("📷 使用攝影機拍照")
    picture = st.camera_input("請啟用攝影機")
    if picture:
        file_bytes = np.frombuffer(picture.getvalue(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        blue, lap, brightness = analyze_image(image)
        result, pm25 = rule_based_prediction(blue, lap, brightness)
        st.image(image, channels="BGR", caption="即時影像", use_column_width=True)
        st.markdown(f"- 藍天比例：{blue:.2f}%")
        st.markdown(f"- 清晰度（Laplacian）：{lap:.2f}")
        st.markdown(f"- 平均亮度：{brightness:.2f}")
        st.info(f"🔍 影像判斷：{result}")
        st.success("✅ PM2.5 狀態：未超標" if pm25 == "未超標" else "❌ PM2.5 狀態：超標" if pm25 == "超標" else "⚠️ PM2.5 狀態：無法判斷")

        save_to_db([{
            "檔名": "camera.jpg",
            "藍天比例(%)": round(blue, 2),
            "Laplacian平均值": round(lap, 2),
            "平均亮度": round(brightness, 2),
            "判斷結果": result,
            "PM2.5狀態": pm25
        }])

with tab2:
    st.subheader("📁 上傳圖片分析")
    uploaded_files = st.file_uploader("上傳多張圖片", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        results = []
        for file in uploaded_files:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            blue, lap, brightness = analyze_image(image)
            result, pm25 = rule_based_prediction(blue, lap, brightness)
            thumbnail = cv2.resize(image, (120, 90))
            thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
            results.append({
                "圖片縮圖": thumbnail,
                "檔名": file.name,
                "藍天比例(%)": round(blue, 2),
                "Laplacian平均值": round(lap, 2),
                "平均亮度": round(brightness, 2),
                "判斷結果": result,
                "PM2.5狀態": pm25
            })

        df_result = pd.DataFrame(results)
        st.subheader("📋 預測結果")
        for row in results:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(row["圖片縮圖"], use_column_width=True)
            with col2:
                st.markdown(f"**檔名**: {row['檔名']}")
                st.markdown(f"- 藍天比例：{row['藍天比例(%)']}%")
                st.markdown(f"- 清晰度（Laplacian）：{row['Laplacian平均值']}")
                st.markdown(f"- 平均亮度：{row['平均亮度']}")
                st.info(f"🔍 影像判斷：{row['判斷結果']}")
                if row["PM2.5狀態"] == "未超標":
                    st.success("PM2.5 狀態：未超標")
                elif row["PM2.5狀態"] == "超標":
                    st.error("PM2.5 狀態：超標")
                else:
                    st.warning("PM2.5 狀態：無法判斷")

        save_to_db(results)

        st.subheader("📊 狀態分佈")
        counts = df_result["PM2.5狀態"].value_counts()
        fig, ax = plt.subplots()
        ax.bar(counts.index, counts.values, color=["green", "red", "blue"])
        ax.set_ylabel("數量")
        ax.set_title("PM2.5 預測分佈")
        st.pyplot(fig)

        csv = df_result.drop(columns=["圖片縮圖"]).to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 下載 CSV", csv, file_name="PM2.5預測.csv")

        pdf_path = "PM2.5_報告.pdf"
        generate_pdf(results, pdf_path)
        with open(pdf_path, "rb") as f:
            st.download_button("🧾 下載 PDF 報告", f.read(), file_name=pdf_path, mime="application/pdf")

with tab3:
    st.subheader("📅 查詢歷史資料")
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("開始日期", value=datetime.now() - timedelta(days=7))
    with col2:
        end = st.date_input("結束日期", value=datetime.now())
    if st.button("🔍 查詢"):
        df = query_history(str(start), str(end + timedelta(days=1)))
        if df.empty:
            st.warning("找不到資料")
        else:
            st.dataframe(df)
            st.subheader("📈 PM2.5 趨勢")
            show_history_chart(df)
