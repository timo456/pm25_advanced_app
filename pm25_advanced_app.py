# ✅ Unified PM2.5 Detection App with ML-Based Logic（完整版）
import streamlit as st
st.set_page_config(page_title="PM2.5 預測系統（機器學習版）", layout="wide")
import pandas as pd
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
import matplotlib.font_manager as fm

# 字型設定
font_path = "fonts/NotoSerifTC-Regular.ttf"
my_font = fm.FontProperties(fname=font_path)
plt.rcParams["axes.unicode_minus"] = False
st.text(f"✅ 目前載入的字型為: {my_font.get_name()}")
st.title("🌫️ PM2.5 空氣品質多功能預測系統（機器學習）")

# 載入模型
model = joblib.load("pm25_model.pkl")

# 特徵提取函數
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

# 模型預測
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

# PDF 匯出
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

# Tabs with full functionality
tab1, tab2, tab3 = st.tabs(["📷 即時預測", "📁 圖片分析", "🧾 產出 PDF 報告"])

# 📷 即時攝影機預測
with tab1:
    st.subheader("📷 使用攝影機拍照")
    picture = st.camera_input("請啟用攝影機")
    if picture:
        file_bytes = np.frombuffer(picture.getvalue(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        blue, lap, lap_std, sobel, brightness = analyze_image(image)
        result, pm25 = ml_model_prediction(blue, lap, lap_std, sobel, brightness)
        st.image(image, channels="BGR", caption="即時影像", use_column_width=True)
        st.markdown(f"- 藍天比例：{blue:.2f}%")
        st.markdown(f"- 清晰度（Laplacian）：{lap:.2f}")
        st.markdown(f"- 標準差（Laplacian）：{lap_std:.2f}")
        st.markdown(f"- Sobel 邊緣強度：{sobel:.2f}")
        st.markdown(f"- 平均亮度：{brightness:.2f}")
        st.info(f"🔍 影像判斷：{result}")
        st.success("✅ PM2.5 狀態：未超標" if pm25 == "未超標" else "❌ PM2.5 狀態：超標" if pm25 == "超標" else "⚠️ PM2.5 狀態：無法判斷")

# 📁 圖片分析上傳
with tab2:
    st.subheader("📁 上傳圖片進行分析")
    uploaded_files = st.file_uploader("上傳圖片 (可多選)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        results = []
        for file in uploaded_files:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            blue, lap, lap_std, sobel, brightness = analyze_image(image)
            result, pm25 = ml_model_prediction(blue, lap, lap_std, sobel, brightness)
            st.image(image, channels="BGR", caption=file.name, use_column_width=True)
            st.markdown(f"- 藍天比例：{blue:.2f}%")
            st.markdown(f"- Laplacian：{lap:.2f}")
            st.markdown(f"- Laplacian 標準差：{lap_std:.2f}")
            st.markdown(f"- Sobel 邊緣強度：{sobel:.2f}")
            st.markdown(f"- 平均亮度：{brightness:.2f}")
            st.info(f"🔍 判斷：{result}")
            st.success("PM2.5 狀態：未超標" if pm25 == "未超標" else "PM2.5 狀態：超標" if pm25 == "超標" else "PM2.5 狀態：無法判斷")
            results.append({
                "檔名": file.name,
                "藍天比例(%)": blue,
                "Laplacian平均值": lap,
                "Laplacian標準差": lap_std,
                "Sobel邊緣強度": sobel,
                "平均亮度": brightness,
                "判斷結果": result,
                "PM2.5狀態": pm25
            })

        # PDF 報表輸出按鈕
        st.markdown("### 📄 匯出報表")
        if results:
            df_result = pd.DataFrame(results)
            csv = df_result.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 下載 CSV", csv, file_name="PM2.5預測結果.csv")

        # 圖表
        st.subheader("📊 狀態分佈圖")
        counts = df_result["PM2.5狀態"].value_counts()

        # 設定顏色對應
        color_map = {
            "未超標": "green",
            "超標": "red",
            "無法判斷": "blue"
        }
        bar_colors = [color_map.get(label, "gray") for label in counts.index]

        fig, ax = plt.subplots()
        ax.bar(counts.index, counts.values, color=bar_colors)
        ax.set_ylabel("數量",fontproperties=my_font)
        ax.set_title("PM2.5 預測統計",fontproperties=my_font)
        ax.set_xticklabels(counts.index, fontproperties=my_font)
        st.pyplot(fig)

        pdf_path = "PM2.5_報告.pdf"
        generate_pdf(results, pdf_path)
        with open(pdf_path, "rb") as f:
            st.download_button("🧾 下載 PDF 報告", f.read(), file_name=pdf_path, mime="application/pdf")

# 🧾 模型說明 or 歷史查詢功能可加在 tab3
with tab3:
    st.subheader("🧾 說明 / 日誌")
    st.write("未來這裡可以加上歷史記錄查詢或模型訓練摘要。")
