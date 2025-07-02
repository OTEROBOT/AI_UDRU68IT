import numpy as np
import pandas as pd

# สร้างข้อมูลตัวอย่าง (หรือจะโหลดจากไฟล์ Kaggle จริงก็ได้)
data = pd.DataFrame({
    'Age': [25, 30, 50],
    'Salary': [30000, 45000, 120000],
    'Score': [50, 80, 90]
})

print("🎯 ข้อมูลก่อน Normalize:")
print(data)

# ใช้ Min-Max Normalization
normalized_data = (data - data.min()) / (data.max() - data.min())

print("\n✅ ข้อมูลหลัง Normalize (Min-Max):")
print(normalized_data)
