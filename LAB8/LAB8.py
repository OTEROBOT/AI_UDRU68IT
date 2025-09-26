# LAB8_Haberman.py

# ----------------------------
# Import ไลบรารีที่ใช้ (ตามบทเรียน)
# ----------------------------
import numpy as np  # (บท 2 การแทนปัญหา) ใช้คำนวณเชิงตัวเลขและเมทริกซ์
import pandas as pd  # (บท 1 ข้อมูลและการเตรียมข้อมูล) จัดการ dataset และสร้างตาราง
import tensorflow as tf  # (บท 10 Neural Network) สำหรับสร้างและฝึก ANN
from sklearn.model_selection import train_test_split, cross_val_score  # (บท 7 การทดสอบโมเดล) สำหรับ train/test และ Cross-Validation
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # (บท 1) การปรับมาตรฐานและการเข้ารหัส Label
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, adjusted_rand_score  # (บท 7) วัดผลการทำงานโมเดล
from sklearn.naive_bayes import GaussianNB  # (บท 5 Naive Bayes) ตัวจำแนกเชิงความน่าจะเป็น
from sklearn.tree import DecisionTreeClassifier  # (บท 4 Decision Tree) ตัวจำแนกต้นไม้
from sklearn.neighbors import KNeighborsClassifier  # (บท 3 & 6 KNN) ตัวจำแนก K-Nearest Neighbors
from sklearn.cluster import KMeans  # (บท 9 Clustering) การจัดกลุ่มข้อมูล K-Means
from sklearn.feature_selection import SelectKBest, f_classif  # (บท 8 Feature Selection) เลือกคุณลักษณะที่ดีที่สุด

# ----------------------------
# ฟังก์ชันช่วย (บท 7 การทดสอบโมเดล)
# ----------------------------
def compute_metrics(y_true, y_pred):  
    # คำนวณตัวชี้วัด Accuracy, Precision, Recall, F1
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1}

def print_metrics_html(method_name, metrics):  
    # แปลงผลลัพธ์ของโมเดลเป็นตาราง HTML
    df = pd.DataFrame([metrics], index=[method_name])
    return df.to_html(float_format="%.4f", classes="table table-striped table-bordered table-hover", justify="center", border=1)

# ----------------------------
# โหลด Dataset Haberman (บท 1 ข้อมูลและการเตรียมข้อมูล)
# ----------------------------
data = pd.read_csv("haberman.csv", header=None)  # โหลด CSV
X = data.iloc[:, :-1].values  # Features (age, year, nodes)
y = data.iloc[:, -1].values   # Label (1 = อยู่รอด, 2 = ไม่รอด)

# แบ่ง train/test 70/30
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Scaling ข้อมูล (Normalize)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One-Hot Encoding สำหรับ ANN (บท 10)
encoder = OneHotEncoder(sparse_output=False)
y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_onehot = encoder.transform(y_test.reshape(-1, 1))

# ----------------------------
# 1–2. Representation & Problem Solving (บท 2)
# ----------------------------
def create_adjacency_matrix(X, k=3):  
    # สร้าง adjacency matrix แทนโครงสร้างความใกล้กัน
    n_samples = X.shape[0]
    adj_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        distances = np.sum((X - X[i])**2, axis=1)  # ระยะ Euclidean
        nearest_indices = np.argsort(distances)[1:k+1]  # หาตัวใกล้สุด k ตัว
        adj_matrix[i, nearest_indices] = 1
    return adj_matrix

adj_matrix = create_adjacency_matrix(X_train_scaled)
adj_html = "<h3>การแทนปัญหาและการแก้ปัญหา (Adjacency Matrix)</h3>"

# ----------------------------
# 8. Feature Selection (บท 8)
# ----------------------------
selector = SelectKBest(score_func=f_classif, k=2)  
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# ----------------------------
# 5. Naive Bayes (บท 5)
# ----------------------------
nb = GaussianNB()
nb.fit(X_train_selected, y_train)
nb_pred = nb.predict(X_test_selected)
nb_metrics = compute_metrics(y_test, nb_pred)

# ----------------------------
# 4. Decision Tree (บท 4)
# ----------------------------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_selected, y_train)
dt_pred = dt.predict(X_test_selected)
dt_metrics = compute_metrics(y_test, dt_pred)

# ----------------------------
# 3 & 6. KNN (บท 3 + 6)
# ----------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_selected, y_train)
knn_pred = knn.predict(X_test_selected)
knn_metrics = compute_metrics(y_test, knn_pred)

# ----------------------------
# 9. K-Means Clustering (บท 9)
# ----------------------------
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train_selected)
cluster_labels = kmeans.predict(X_test_selected)
cluster_score = adjusted_rand_score(y_test, cluster_labels)
cluster_html = f"<h3>Clustering (K-Means)</h3><p>Adjusted Rand Score: {cluster_score:.4f}</p>"

# ----------------------------
# 10. ANN (บท 10 Neural Network)
# ----------------------------
ann = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),  # input 2 features
    tf.keras.layers.Dense(10, activation="relu"),  # hidden layer
    tf.keras.layers.Dense(2, activation="softmax")  # output layer
])
ann.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
ann.fit(X_train_selected, y_train_onehot, epochs=100, batch_size=5, verbose=0)

ann_pred_onehot = ann.predict(X_test_selected)
ann_pred = np.argmax(ann_pred_onehot, axis=1) + 1  # บวก 1 เพื่อให้ label ตรงกับ dataset
ann_metrics = compute_metrics(y_test, ann_pred)

# ----------------------------
# 7. Cross Validation (บท 7)
# ----------------------------
cv_scores = cross_val_score(DecisionTreeClassifier(random_state=42), X_train_selected, y_train, cv=5)
cv_html = f"<h3>การทดสอบโมเดล (Cross-Validation บน Decision Tree)</h3><p>Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})</p>"

# ----------------------------
# HTML Output (ธีมท้องฟ้ายามค่ำคืน)
# ----------------------------
html_output = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ผลลัพธ์การเปรียบเทียบวิธี AI (Haberman Dataset)</title>
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
<style>
  body {
      padding: 20px;
      background: linear-gradient(to bottom, #0f2027, #203a43, #2c5364); /* ท้องฟ้ายามค่ำคืน */
      color: #f1f1f1;
      font-family: "Segoe UI", sans-serif;
  }
  h1, h2, h3 {
      color: #ffdd57; /* สีเหลืองทองให้อ่านง่าย */
      text-shadow: 2px 2px 5px rgba(0,0,0,0.8);
  }
  .table {
      margin: auto;
      width: 85%;
      background: rgba(255,255,255,0.05);
      color: #fff;
  }
  .table th {
      background-color: rgba(0, 0, 50, 0.7);
      color: #ffdd57;
  }
  .table td {
      background-color: rgba(255, 255, 255, 0.05);
  }
  .container {
      background: rgba(0,0,0,0.4);
      padding: 20px;
      border-radius: 15px;
      box-shadow: 0 0 30px rgba(0,0,0,0.7);
  }
</style>
</head>
<body>
<div class="container">
<h1 class="text-center">🌌 ผลลัพธ์การเปรียบเทียบวิธี AI (Haberman Dataset) 🌌</h1>
<h2>🔮 วิธีการจำแนก</h2>
"""
html_output += print_metrics_html("Naive Bayes", nb_metrics)
html_output += print_metrics_html("Decision Tree", dt_metrics)
html_output += print_metrics_html("KNN", knn_metrics)
html_output += print_metrics_html("ANN", ann_metrics)

html_output += adj_html
html_output += cluster_html
html_output += cv_html
html_output += "</div></body></html>"

with open("haberman_results.html", "w", encoding="utf-8") as f:
    f.write(html_output)

print("✅ บันทึกผลลัพธ์ HTML ไปที่ 'haberman_results.html' เปิดดูในเบราว์เซอร์ได้เลย")
