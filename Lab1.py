import pandas as pd

# โหลดข้อมูลจาก fruit.csv
data = pd.read_csv('fruit.csv')

# แสดงทุกคอลัมน์ทั้งหมดในตาราง
print("📋 ข้อมูลทั้งหมด (ทุกคอลัมน์):")
print(data.to_string(index=False))

# แสดงจำนวนแถวและคอลัมน์
num_rows, num_cols = data.shape
print(f"\n📊 มีทั้งหมด {num_rows} แถว และ {num_cols} คอลัมน์")

# -------------------------------
# 🎯 แสดงข้อมูลบางส่วนก่อน Normalize
# -------------------------------

print("\n🎯 ข้อมูลก่อน Normalize (เฉพาะ features ที่จะ Normalize):")
print(data[['fruit_name', 'mass', 'width', 'height', 'color_score']].head())

# -------------------------------
# 🔧 เลือกเฉพาะคอลัมน์ที่จะ Normalize
# -------------------------------

features = data[['mass', 'width', 'height', 'color_score']]

# -------------------------------
# 🔁 Min-Max Normalize
# -------------------------------

normalized_data = (features - features.min()) / (features.max() - features.min())

# เพิ่ม fruit_name กลับมาเพื่อดูว่าแถวไหนคือผลไม้อะไร
normalized_data['fruit_name'] = data['fruit_name']

# -------------------------------
# ✅ แสดงข้อมูลหลัง Normalize (ทุกแถว)
# -------------------------------

print("\n✅ ข้อมูลหลัง Min-Max Normalize:")
print(normalized_data.to_string(index=False))

# แสดงจำนวนแถวและคอลัมน์ของ normalized data
n_rows, n_cols = normalized_data.shape
print(f"\n📐 ตาราง Normalize มี {n_rows} แถว และ {n_cols} คอลัมน์")
