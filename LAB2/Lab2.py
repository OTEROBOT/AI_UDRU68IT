import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# โหลดข้อมูลจากไฟล์ CSV
df = pd.read_csv("Iris.csv")  # ไฟล์ต้องอยู่ในโฟลเดอร์เดียวกัน

# ลบคอลัมน์ Id ทิ้ง เพราะไม่ใช่ Feature
df = df.drop(columns=["Id"])

# แยก Feature และ Label
X = df.drop(columns=["Species"])
y = df["Species"]

# แบ่งข้อมูลเป็นชุด Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# สร้างโมเดลต้นไม้ตัดสินใจ
clf = DecisionTreeClassifier(criterion="entropy")
clf.fit(X_train, y_train)

# แสดงต้นไม้
plt.figure(figsize=(10, 6))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=clf.classes_)
plt.title("🌳 ต้นไม้ตัดสินใจจากไฟล์ Iris.csv")
plt.show()

# ความแม่นยำ
accuracy = clf.score(X_test, y_test)
print(f"🎯 ความแม่นยำของโมเดล: {accuracy*100:.2f}%")

# ทำนายข้อมูลใหม่
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = clf.predict(sample)
print(f"\n🔮 ทำนายจากตัวอย่าง {sample} → {prediction[0]}")

# แสดงกฎแบบ if-then
rules = export_text(clf, feature_names=list(X.columns))
print("\n📜 กฎจากต้นไม้ตัดสินใจ:")
print(rules)
