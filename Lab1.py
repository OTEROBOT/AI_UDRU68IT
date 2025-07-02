import numpy as np
import pandas as pd

# โหลดข้อมูลจากไฟล์ fruit.csv
data = pd.read_csv('fruit.csv')

# แสดงข้อมูลบางส่วนก่อน Normalize
print("🎯 ข้อมูลก่อน Normalize:")
print(data[['fruit_name', 'mass', 'width', 'height', 'color_score']].head())

# เลือกเฉพาะ columns ที่ต้อง Normalize
features = data[['mass', 'width', 'height', 'color_score']]

# ใช้ Min-Max Normalization
normalized_data = (features - features.min()) / (features.max() - features.min())

# รวมกับชื่อผลไม้ เพื่อดูว่าแต่ละแถวคือผลไม้อะไร
normalized_data['fruit_name'] = data['fruit_name']

# แสดงผลลัพธ์
print("\n✅ ข้อมูลหลัง Normalize (Min-Max):")
print(normalized_data.head())
