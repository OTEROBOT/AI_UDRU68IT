import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. โหลดข้อมูลจาก CSV
df = pd.read_csv('winequality-red.csv', sep=';')
print("📋 ข้อมูลตัวอย่าง:")
print(df.head(), "\n")
print("📊 กระจายคะแนนคุณภาพ:")
print(df['quality'].value_counts().sort_index(), "\n")

# 2. จัด label ให้เป็น classification แบบง่าย
#    แบ่งคุณภาพ >=7 = 'good', else 'not good'
df['quality_label'] = df['quality'].apply(lambda x: 'good' if x >= 7 else 'not good')

# 3. แยก feature และ label
X = df.drop(columns=['quality', 'quality_label'])
y = df['quality_label']

# 4. แบ่งชุด train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# 5. ฝึกโมเดล Decision Tree
clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
clf.fit(X_train, y_train)

# 6. วัดความแม่นยำ
acc = clf.score(X_test, y_test)
print(f"🎯 Accuracy ของโมเดล: {acc*100:.2f}%\n")

# 7. แสดงต้นไม้
plt.figure(figsize=(16,10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=clf.classes_)
plt.title('🌳 Decision Tree: Wine Quality (red wine)')
plt.show()

# 8. แสดงกฎแบบ if-then
rules = export_text(clf, feature_names=list(X.columns))
print("📜 กฎจากต้นไม้:")
print(rules)

# 9. ทำนายจากข้อมูลใหม่ (ตัวอย่างแถวแรก)
sample = X.iloc[[0]]
pred = clf.predict(sample)
print(f"\n🔮 ตัวอย่างแรก {sample.values.tolist()[0]} → ทำนาย: {pred[0]}")
