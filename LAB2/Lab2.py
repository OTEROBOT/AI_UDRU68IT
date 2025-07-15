import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# ฟังก์ชันสำหรับรับข้อมูลจากผู้ใช้
def get_user_input():
    print("\n🧪 === เริ่มกรอกข้อมูลผู้ป่วยเพื่อทำนายโรคเบาหวาน ===")
    print("📝 คำแนะนำ: กรอกตัวเลขที่เหมาะสมตามคำอธิบาย ไม่ใช้ตัวอักษรหรือค่าว่าง")
    print("   - Pregnancies: จำนวนครั้งที่ตั้งครรภ์ (เช่น 0-10)")
    print("   - Glucose: ระดับกลูโคสในเลือด (เช่น 0-200 mg/dL)")
    print("   - BloodPressure: ความดันโลหิต (เช่น 0-120 mm Hg)")
    print("   - SkinThickness: ความหนาของผิวหนัง (เช่น 0-50 mm)")
    print("   - Insulin: ระดับอินซูลิน (เช่น 0-500 mu U/ml)")
    print("   - BMI: ดัชนีมวลกาย (เช่น 0-50 kg/m²)")
    print("   - DiabetesPedigreeFunction: ความเสี่ยงทางพันธุกรรม (เช่น 0.0-2.0)")
    print("   - Age: อายุ (เช่น 20-80 ปี)\n")
    
    test_data = {}
    fields = [
        ('Pregnancies', 'จำนวนครั้งที่ตั้งครรภ์: ', int),
        ('Glucose', 'ระดับกลูโคสในเลือด: ', float),
        ('BloodPressure', 'ความดันโลหิต (mm Hg): ', float),
        ('SkinThickness', 'ความหนาของผิวหนัง (mm): ', float),
        ('Insulin', 'ระดับอินซูลิน (mu U/ml): ', float),
        ('BMI', 'ดัชนีมวลกาย (kg/m²): ', float),
        ('DiabetesPedigreeFunction', 'คะแนนความเสี่ยงทางพันธุกรรม: ', float),
        ('Age', 'อายุ (ปี): ', int)
    ]
    
    for field, prompt, dtype in fields:
        while True:
            try:
                value = input(prompt).strip()
                if not value:
                    print("❌ กรุณากรอกตัวเลข ไม่สามารถเว้นว่างได้")
                    continue
                value = dtype(value)
                if value < 0:
                    print("❌ ค่าต้องไม่เป็นลบ กรุณากรอกใหม่")
                    continue
                test_data[field] = value
                break
            except ValueError:
                print(f"❌ กรุณากรอกตัวเลขที่ถูกต้องสำหรับ {field} (เช่น { '0 หรือ 1' if dtype == int else '120.0 หรือ 0.5' })")
    
    return test_data

# 1. ตรวจสอบและโหลดข้อมูล (เลือก 20 แถวแรก)
file_path = 'diabetes.csv'  # ปรับพาธถ้าจำเป็น
try:
    df = pd.read_csv(file_path).head(20)  # จำกัด 20 แถว
    print("📋 ข้อมูลตัวอย่าง (20 แถวแรก):")
    print(df.head(), "\n")
    print("📊 กระจายของ Outcome (0=ไม่มีเบาหวาน, 1=มีเบาหวาน):")
    print(df['Outcome'].value_counts(), "\n")
except FileNotFoundError:
    print(f"❌ ไม่พบไฟล์ {file_path} กรุณาดาวน์โหลด diabetes.csv จาก https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database และวางใน {os.getcwd()}")
    print("⚠️ กรุณาวางไฟล์ในโฟลเดอร์ที่ถูกต้องแล้วรันโค้ดใหม่")
    exit(1)
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการโหลดข้อมูล: {str(e)}")
    exit(1)

# 2. จัดการข้อมูล (แทนที่ค่า 0 ในบางคอลัมน์ด้วยค่าเฉลี่ย)
columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns_to_replace:
    df[col] = df[col].replace(0, df[col][df[col] != 0].mean())

# 3. แยก feature และ label
try:
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการแยกข้อมูล: {str(e)}")
    exit(1)

# 4. แบ่งชุด train/test
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการแบ่งข้อมูล: {str(e)}")
    exit(1)

# 5. ฝึกโมเดล Decision Tree
try:
    clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
    clf.fit(X_train, y_train)
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการฝึกโมเดล: {str(e)}")
    exit(1)

# 6. รับข้อมูลจากผู้ใช้และทำนาย (ก่อนแสดงผลอื่นๆ)
print("\n✅ โค้ดเริ่มส่วนกรอกข้อมูลเพื่อทำนาย")
try:
    while True:
        test_data_user = get_user_input()
        test_df_user = pd.DataFrame([test_data_user])
        pred_user = clf.predict(test_df_user)
        print(f"\n🔮 ข้อมูลที่คุณกรอก {test_data_user} → ทำนาย: {'Diabetes' if pred_user[0] == 1 else 'No Diabetes'}")
        
        # ถามว่าต้องการกรอกข้อมูลใหม่หรือไม่
        while True:
            again = input("\n🎯 ต้องการกรอกข้อมูลใหม่เพื่อทำนายอีกครั้งหรือไม่? (y/n): ").lower().strip()
            if again in ['y', 'n']:
                break
            print("❌ กรุณากรอก 'y' หรือ 'n' เท่านั้น")
        if again == 'n':
            print("✅ สิ้นสุดการกรอกข้อมูล ไปยังส่วนแสดงผลลัพธ์")
            break
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการกรอกข้อมูลหรือทำนาย: {str(e)}")
    exit(1)

# 7. วัดความแม่นยำ
try:
    acc = clf.score(X_test, y_test)
    print(f"\n🎯 Accuracy ของโมเดล: {acc*100:.2f}%")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการคำนวณความแม่นยำ: {str(e)}")

# 8. แสดงต้นไม้
try:
    plt.figure(figsize=(12, 8))
    plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'])
    plt.title('🌳 Decision Tree: Pima Indians Diabetes Classification')
    plt.savefig('diabetes_decision_tree.png')  # บันทึกรูปภาพ
    plt.show()
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการแสดงต้นไม้: {str(e)}")

# 9. แสดงกฎแบบ if-then
try:
    rules = export_text(clf, feature_names=list(X.columns))
    print("\n📜 กฎจากต้นไม้:")
    print(rules)
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการแสดงกฎ: {str(e)}")

# 10. ทำนายจากข้อมูลตัวอย่าง (แถวแรก)
try:
    sample = X.iloc[[0]]
    pred = clf.predict(sample)
    print(f"\n🔮 ตัวอย่างแรก {sample.values.tolist()[0]} → ทำนาย: {'Diabetes' if pred[0] == 1 else 'No Diabetes'}")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการทำนายตัวอย่าง: {str(e)}")

# 11. ตัวอย่างการทดสอบด้วยข้อมูลที่กำหนดไว้
try:
    print("\n🧪 ทดสอบด้วยข้อมูลที่กำหนดไว้:")
    test_data = {
        'Pregnancies': 2,
        'Glucose': 120.0,
        'BloodPressure': 70.0,
        'SkinThickness': 20.0,
        'Insulin': 80.0,
        'BMI': 32.0,
        'DiabetesPedigreeFunction': 0.5,
        'Age': 30
    }
    test_df = pd.DataFrame([test_data])
    pred_test = clf.predict(test_df)
    print(f"ข้อมูลทดสอบ {test_data} → ทำนาย: {'Diabetes' if pred_test[0] == 1 else 'No Diabetes'}")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการทำนายข้อมูลที่กำหนด: {str(e)}")

print("\n✅ โค้ดทำงานเสร็จสิ้น ขอบคุณที่ใช้งาน!")