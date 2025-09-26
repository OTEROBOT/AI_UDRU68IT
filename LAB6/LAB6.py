# ================================================================
# คู่มือการรัน Streamlit สำหรับการวิเคราะห์ Titanic Dataset
# ================================================================
# 1. ตรวจสอบไฟล์ Dataset
#    - โค้ดนี้ต้องใช้ไฟล์ "Titanic-Dataset.csv" อยู่ในโฟลเดอร์เดียวกับไฟล์ .py
#    - หากไฟล์ไม่อยู่ โค้ดจะแสดงข้อความ error และหยุดการทำงาน
#      (ดู os.getcwd() เพื่อเช็คโฟลเดอร์ปัจจุบัน)

# 2. วิธีติดตั้งไลบรารีที่ต้องใช้ (ถ้ายังไม่ติดตั้ง)
#    - เปิด Command Prompt / Terminal แล้วรันคำสั่ง:
#      pip install streamlit pandas plotly numpy
#      (ใช้คำสั่งนี้ใน environment ที่ใช้รันโค้ด เช่น .venv ของโครงการ)

# 3. วิธีรัน Streamlit
#    - เปิด Command Prompt / Terminal
#    - cd ไปที่โฟลเดอร์ที่เก็บไฟล์ .py
#      ตัวอย่าง:
#      cd C:\xampp\htdocs\2567\GitHub_OTEROBOT\AI_UDRU68IT\LAB6
#    - รันคำสั่ง:
#      streamlit run LAB6.py
#    - เบราว์เซอร์จะเปิดขึ้นมาโดยอัตโนมัติ หรือถ้าไม่เปิด
#      ให้คัดลอก URL ที่ Streamlit แสดง เช่น
#      http://localhost:8501
#      ไปวางในเบราว์เซอร์

# ================================================================
# คำอธิบายโค้ดแต่ละส่วน
# ================================================================

# import ไลบรารีที่จำเป็น
import streamlit as st    # สำหรับสร้าง Web app แบบ interactive
import pandas as pd       # สำหรับอ่านและจัดการข้อมูล CSV
import plotly.express as px  # สำหรับสร้างกราฟ interactive
import numpy as np        # สำหรับคำนวณทางคณิตศาสตร์
import os                 # ตรวจสอบไฟล์และ path

# ตั้งค่า Streamlit
st.set_page_config(page_title="การวิเคราะห์ Titanic Dataset", layout="wide")

# ตรวจสอบว่ามีไฟล์ CSV หรือไม่
csv_file = "Titanic-Dataset.csv"
if not os.path.exists(csv_file):
    st.error(f"ไม่พบไฟล์ {csv_file} ในโฟลเดอร์นี้ กรุณาวางไฟล์ใน {os.getcwd()}")
    st.stop()  # หยุดการทำงานถ้าไฟล์ไม่มี

# ฟังก์ชันโหลดข้อมูลและแคชผลลัพธ์เพื่อให้รันเร็วขึ้น
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(csv_file)  # อ่านไฟล์ CSV
        return df
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์: {e}")
        st.stop()

# โหลดข้อมูล
df = load_data()  # ถ้าไฟล์อ่านไม่สำเร็จ df จะเป็น None → error "NoneType not subscriptable"

# เลือกฟีเจอร์ตัวเลขสำหรับวิเคราะห์ และลบค่า NaN
features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
df_clean = df[features + ["Survived"]].dropna()

# ================================================================
# คำอธิบายเพิ่มเติม
# - df[features + ["Survived"]] คือการเลือกเฉพาะคอลัมน์ที่สนใจ
# - .dropna() คือการลบแถวที่มีค่า NaN เพื่อให้คำนวณค่า mean/variance ได้ถูกต้อง
# ================================================================

# ข้อมูลทั่วไปของ Dataset
total_samples = len(df_clean)  # จำนวนตัวอย่างทั้งหมด
num_features = len(features)    # จำนวนฟีเจอร์
classes = df_clean["Survived"].nunique()  # จำนวนคลาส (รอด/ไม่รอด)
class_distribution = df_clean["Survived"].value_counts().sort_index().tolist()  # กระจายคลาส

# แยกข้อมูลตามคลาส
class0 = df_clean[df_clean["Survived"] == 0]  # ไม่รอด
class1 = df_clean[df_clean["Survived"] == 1]  # รอด
class0_samples = len(class0)
class1_samples = len(class1)

# คำนวณ mean และ variance ของแต่ละฟีเจอร์ในแต่ละคลาส
mean0 = class0[features].mean()
mean1 = class1[features].mean()
var0 = class0[features].var(ddof=1)
var1 = class1[features].var(ddof=1)

# คำนวณ Distance ตามสูตร
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

# สร้าง DataFrame สำหรับแสดงผล
df_result = pd.DataFrame(list(results.items()), columns=["Feature", "Distance"])
df_result = df_result.sort_values(by="Distance", ascending=False)

# ================================================================
# สรุป:
# - ใช้ Streamlit เพื่อสร้าง Web App
# - วิเคราะห์ Titanic Dataset ว่าแต่ละฟีเจอร์มี "Distance" ระหว่างรอด vs ไม่รอดเท่าไหร่
# - สามารถดูค่า mean, variance และกราฟแท่งเปรียบเทียบฟีเจอร์ได้
# ================================================================
