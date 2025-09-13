import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os

# ตั้งค่า Streamlit
st.set_page_config(page_title="การวิเคราะห์ Titanic Dataset", layout="wide")

# ตรวจสอบว่าไฟล์ Titanic-Dataset.csv มีอยู่หรือไม่
csv_file = "Titanic-Dataset.csv"
if not os.path.exists(csv_file):
    st.error(f"ไม่พบไฟล์ {csv_file} ในโฟลเดอร์นี้ กรุณาวางไฟล์ใน {os.getcwd()}")
    st.stop()

# โหลดข้อมูล Titanic Dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์: {e}")
        st.stop()

df = load_data()

# เลือกฟีเจอร์ตัวเลข
features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
df_clean = df[features + ["Survived"]].dropna()

# ข้อมูลทั่วไปของ Dataset
total_samples = len(df_clean)
num_features = len(features)
classes = df_clean["Survived"].nunique()
class_distribution = df_clean["Survived"].value_counts().sort_index().tolist()

class0 = df_clean[df_clean["Survived"] == 0]  # ไม่รอด
class1 = df_clean[df_clean["Survived"] == 1]  # รอด
class0_samples = len(class0)
class1_samples = len(class1)

# คำนวณ mean และ variance
mean0 = class0[features].mean()
mean1 = class1[features].mean()
var0 = class0[features].var(ddof=1)
var1 = class1[features].var(ddof=1)

# คำนวณ Distance
results = {}
for feature in features:
    mu_i, mu_j = mean0[feature], mean1[feature]
    var_i, var_j = var0[feature], var1[feature]
    try:
        # สูตร Distance: d_ij = (1/2)(σ²j/σ²i + σ²i/σ²j - 2) + (1/2)(μi - μj)²(1/σ²i + 1/σ²j)
        d = 0.5 * ((var_j / var_i) + (var_i / var_j) - 2) + 0.5 * ((mu_i - mu_j) ** 2) * ((1 / var_i) + (1 / var_j))
        results[feature] = d
    except ZeroDivisionError:
        st.warning(f"ไม่สามารถคำนวณ Distance สำหรับฟีเจอร์ {feature} เนื่องจาก variance เป็นศูนย์")
        results[feature] = 0.0

# สร้าง DataFrame สำหรับผลลัพธ์
df_result = pd.DataFrame(list(results.items()), columns=["Feature", "Distance"])
df_result = df_result.sort_values(by="Distance", ascending=False)

# ส่วนการแสดงผลใน Streamlit
st.title("การวิเคราะห์ Titanic Dataset", anchor=False)

# ส่วนข้อมูล Dataset
st.header("=== ข้อมูล Titanic Dataset ===", anchor=False)
col1, col2, col3, col4 = st.columns(4)
col1.metric("จำนวนตัวอย่างทั้งหมด", total_samples)
col2.metric("จำนวนฟีเจอร์", num_features)
col3.metric("จำนวนคลาส", classes)
col4.metric("การกระจายของคลาส", str(class_distribution))

# ส่วนการวิเคราะห์ Class 0 vs Class 1
st.header("=== การวิเคราะห์ Class 0 vs Class 1 ===", anchor=False)
col1, col2 = st.columns(2)
col1.metric("จำนวนตัวอย่าง Class 0 (ไม่รอด)", class0_samples)
col2.metric("จำนวนตัวอย่าง Class 1 (รอด)", class1_samples)

# ส่วนการคำนวณ Distance
st.header("=== การคำนวณ Distance โดยใช้สูตรที่กำหนด ===", anchor=False)
st.write("สูตร: d_ij = (1/2)(σ²j/σ²i + σ²i/σ²j - 2) + (1/2)(μi - μj)²(1/σ²i + 1/σ²j)")
st.write("โดยที่ i = Class 0, j = Class 1")

st.subheader("การวิเคราะห์ฟีเจอร์ทั้งหมด", anchor=False)
for idx, row in df_result.iterrows():
    feature = row["Feature"]
    distance = row["Distance"]
    st.markdown(f"**{idx + 1}. {feature}:**")
    st.markdown(f"   Class 0: μ = {mean0[feature]:.3f}, σ² = {var0[feature]:.3f}")
    st.markdown(f"   Class 1: μ = {mean1[feature]:.3f}, σ² = {var1[feature]:.3f}")
    st.markdown(f"   Distance: {distance:.6f}")

# ตารางสรุป
st.header("=== ตารางสรุป (ฟีเจอร์ทั้งหมด) ===", anchor=False)
st.dataframe(df_result.style.format({"Distance": "{:.6f}"}), use_container_width=True)

# ผลลัพธ์และอันดับ Top 5
st.header("=== ผลลัพธ์ ===", anchor=False)
best_feature = df_result.iloc[0]["Feature"]
best_distance = df_result.iloc[0]["Distance"]
st.success(f"**ฟีเจอร์ที่ดีที่สุด:** {best_feature}")
st.success(f"**คะแนน Distance:** {best_distance:.6f}")

st.subheader("อันดับ Top 5 ฟีเจอร์", anchor=False)
for idx in range(min(5, len(df_result))):
    feature = df_result.iloc[idx]["Feature"]
    distance = df_result.iloc[idx]["Distance"]
    st.markdown(f"{idx + 1}. {feature}: {distance:.6f}")

# กราฟแท่งแสดง Distance
st.header("=== กราฟแสดง Distance ของฟีเจอร์ ===", anchor=False)
fig = px.bar(df_result, x="Feature", y="Distance", title="Distance ของแต่ละฟีเจอร์",
             labels={"Feature": "ฟีเจอร์", "Distance": "คะแนน Distance"},
             color="Distance", color_continuous_scale="Viridis",
             text="Distance")
fig.update_traces(texttemplate="%{text:.6f}", textposition="auto")
fig.update_layout(showlegend=False, title_x=0.5)
st.plotly_chart(fig, use_container_width=True)