🚢 Titanic Analysis Dashboard (Night Sky Theme)

แดชบอร์ดวิเคราะห์ Titanic Dataset ที่ออกแบบด้วย Streamlit ตกแต่งธีม
Beautiful Night Sky พร้อมกราฟสวย ๆ จาก Plotly

  ------------------------
  📦 การติดตั้งเบื้องต้น
  ------------------------

1.  ติดตั้ง Python 3.9+
    ดาวน์โหลดได้จาก: https://www.python.org/downloads/

2.  โคลนโปรเจกต์จาก GitHub

        git clone https://github.com/USERNAME/Titanic-Analysis.git
        cd Titanic-Analysis

3.  ติดตั้งไลบรารีที่จำเป็น

        pip install -r requirements.txt

    👉 ถ้าไม่มีไฟล์ requirements.txt ให้ติดตั้งเองด้วยคำสั่งนี้:

        pip install streamlit pandas plotly numpy

  ------------------
  📂 โครงสร้างไฟล์
  ------------------

Titanic-Analysis/ 
│ 
├── LAB6.py # ไฟล์หลักสำหรับรัน Streamlit Dashboard
├── Titanic-Dataset.csv # ไฟล์ dataset (ต้องวางไว้ในโฟลเดอร์เดียวกับLAB6.py) 
├── requirements.txt # รายการไลบรารีที่ต้องติดตั้ง 
└── README.txt # คู่มือการใช้งาน (ไฟล์นี้)

  ------------------
  🚀 วิธีการใช้งาน
  ------------------

1.  ตรวจสอบให้แน่ใจว่ามี Titanic-Dataset.csv

    -   วางไฟล์ Titanic-Dataset.csv ไว้ในโฟลเดอร์เดียวกับ LAB6.py
    -   ถ้าไม่มีไฟล์ dataset สามารถดาวน์โหลดได้จาก:
        https://www.kaggle.com/c/titanic/data

2.  รันแอปพลิเคชันด้วย Streamlit

        streamlit run LAB6.py

3.  จะต้องมีไฟล์ requirements.txt ในโปรเจคด้วย

4.     คำสั่งสำหรับรันคือ : streamlit run LAB6.py

5.  เปิดใช้งานในเว็บเบราว์เซอร์
    Streamlit จะเปิดอัตโนมัติที่ : http://localhost:8501

  -------------------------
  ✨ ฟีเจอร์ของ Dashboard
  -------------------------

-   สรุปข้อมูล Dataset เบื้องต้น
-   การวิเคราะห์การรอดชีวิตของผู้โดยสาร
-   กราฟ Pie Chart, Bar Chart สวย ๆ จาก Plotly
-   คำนวณ Distance Score ของแต่ละฟีเจอร์
-   UI ธีมท้องฟ้ายามค่ำคืน พร้อมเอฟเฟกต์ดวงดาว 🌌

  -------------
  💡 หมายเหตุ
  -------------

-   ถ้าโค้ดมี error ว่า ไม่พบ Titanic-Dataset.csv → ต้องวางไฟล์ .csv
    ไว้ในโฟลเดอร์เดียวกับ LAB6.py
-   สามารถ deploy ไปที่ Streamlit Cloud ได้เลย
    โดยไม่ต้องแก้โค้ด แค่ upload repo ทั้งหมดขึ้น GitHub

------------------------------------------------------------------------
