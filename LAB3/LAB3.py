
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# โหลดชุดข้อมูล Wine Quality จากไฟล์ท้องถิ่น (ต้องวาง winequality-red.csv ในโฟลเดอร์นี้)
try:
    data = pd.read_csv('winequality-red.csv', sep=',')
    print("โหลดไฟล์สำเร็จ!\n")
    print("ข้อมูลตัวอย่าง 5 แถวแรก:\n")
    print(data.head().to_string(index=False))
except FileNotFoundError:
    print("ข้อผิดพลาด: ไม่พบไฟล์ 'winequality-red.csv' กรุณาดาวน์โหลดจาก https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009 และวางในโฟลเดอร์นี้")
    exit()


# ตรวจสอบว่าคอลัมน์ 'quality' อยู่ในข้อมูล
if 'quality' not in data.columns:
    print("ไม่พบคอลัมน์ 'quality' กรุณาตรวจสอบไฟล์ CSV")
    exit()

# แปลงคะแนนคุณภาพเป็นการจำแนกแบบไบนารี (good: >=7, bad: <7)
data['quality'] = data['quality'].apply(lambda x: 'good' if x >= 7 else 'bad')

# เตรียมข้อมูลคุณสมบัติ (X) และเป้าหมาย (y)
X = data.drop('quality', axis=1)
y = data['quality']

# แบ่งข้อมูลเป็นชุดฝึกสอนและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างและฝึกสอนโมเดล Gaussian Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# ทำนายผลสำหรับชุดทดสอบ
predicted_labels = nb_model.predict(X_test)

# ประเมินความแม่นยำของโมเดล
accuracy = accuracy_score(y_test, predicted_labels)
print('ความแม่นยำของโมเดล:', accuracy)

# อนุญาตให้ผู้ใช้กรอกข้อมูลทดสอบ
print("กรุณากรอกข้อมูลสำหรับการทำนาย (ใส่ค่าตัวเลขตามลำดับ):")
fixed_acidity = float(input("fixed acidity: "))
volatile_acidity = float(input("volatile acidity: "))
citric_acid = float(input("citric acid: "))
residual_sugar = float(input("residual sugar: "))
chlorides = float(input("chlorides: "))
free_sulfur_dioxide = float(input("free sulfur dioxide: "))
total_sulfur_dioxide = float(input("total sulfur dioxide: "))
density = float(input("density: "))
pH = float(input("pH: "))
sulphates = float(input("sulphates: "))
alcohol = float(input("alcohol: "))

# สร้างอาร์เรย์ข้อมูลที่ผู้ใช้ป้อน
user_input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                            free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])

# ทำนายผลสำหรับข้อมูลที่ผู้ใช้ป้อน
user_input_df = pd.DataFrame(user_input_data, columns=X.columns)
predicted_user_label = nb_model.predict(user_input_df)

print('ผลการทำนายสำหรับข้อมูลที่คุณกรอก:', predicted_user_label[0])

# ลิงก์ชุดข้อมูล
print('ลิงก์ชุดข้อมูล:', 'https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009')
