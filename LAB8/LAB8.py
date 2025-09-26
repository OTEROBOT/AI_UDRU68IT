# LAB8_Haberman.py

# ----------------------------
# Import ไลบรารีที่ใช้ (ตามบทเรียน)
# ----------------------------
import numpy as np  # (บท 2 การแทนปัญหา) ใช้สำหรับคำนวณเชิงตัวเลข, เวกเตอร์, เมทริกซ์
import pandas as pd  # (บท 1 ข้อมูลและการเตรียมข้อมูล) ใช้สำหรับโหลดและจัดการ dataset, สร้างตาราง
import tensorflow as tf  # (บท 10 Neural Network) ใช้สร้างและฝึก Artificial Neural Network (ANN)
from sklearn.model_selection import train_test_split, cross_val_score  # (บท 7 การทดสอบโมเดล) แบ่ง train/test และทำ Cross-Validation
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # (บท 1) StandardScaler ปรับค่า feature ให้อยู่ในมาตรฐานเดียวกัน, OneHot แปลง label
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, adjusted_rand_score  # (บท 7) ฟังก์ชันวัดผลโมเดล
from sklearn.naive_bayes import GaussianNB  # (บท 5 Naive Bayes) โมเดลความน่าจะเป็น
from sklearn.tree import DecisionTreeClassifier  # (บท 4 Decision Tree) โมเดลต้นไม้ตัดสินใจ
from sklearn.neighbors import KNeighborsClassifier  # (บท 3 + 6 KNN) โมเดลหาเพื่อนบ้านที่ใกล้ที่สุด
from sklearn.cluster import KMeans  # (บท 9 Clustering) การจัดกลุ่ม K-Means
from sklearn.feature_selection import SelectKBest, f_classif  # (บท 8 Feature Selection) เลือก feature ที่ดีที่สุดด้วย ANOVA F-test

# ----------------------------
# ฟังก์ชันช่วย (บท 7 การทดสอบโมเดล)
# ----------------------------
def compute_metrics(y_true, y_pred):  
    # ฟังก์ชันนี้รับค่าจริง (y_true) และค่าที่โมเดลทำนาย (y_pred)
    # คำนวณตัวชี้วัด 4 ตัว: Accuracy, Precision, Recall, F1
    acc = accuracy_score(y_true, y_pred)  # Accuracy = จำนวนทายถูก / จำนวนทั้งหมด
    prec = precision_score(y_true, y_pred, average="macro")  # Precision = ทายว่า "ใช่" แล้วถูกกี่ %
    rec = recall_score(y_true, y_pred, average="macro")  # Recall = ของจริงทั้งหมด โมเดลทายเจอครบกี่ %
    f1 = f1_score(y_true, y_pred, average="macro")  # F1 = ค่ากลางของ Precision + Recall
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1}  # คืนค่าเป็น dictionary

def print_metrics_html(method_name, metrics):  
    # ฟังก์ชันนี้รับชื่อวิธี (method_name) + ผลลัพธ์ metric
    # สร้าง DataFrame (ตาราง) แล้วแปลงเป็น HTML เพื่อโชว์บนเว็บ
    df = pd.DataFrame([metrics], index=[method_name])  
    return df.to_html(float_format="%.4f", classes="table table-striped table-bordered table-hover", justify="center", border=1)

# ----------------------------
# โหลด Dataset Haberman (บท 1 ข้อมูลและการเตรียมข้อมูล)
# ----------------------------
data = pd.read_csv("haberman.csv", header=None)  # โหลดไฟล์ haberman.csv (ไม่มี header)
X = data.iloc[:, :-1].values  # Features = คอลัมน์แรกถึงก่อนสุดท้าย (3 ตัว: อายุ, ปีผ่าตัด, จำนวนต่อมน้ำเหลือง)
y = data.iloc[:, -1].values   # Label = คอลัมน์สุดท้าย (1 = อยู่รอด, 2 = ไม่รอด)

# แบ่งข้อมูล train/test → 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)  # stratify=y เพื่อให้ class 1/2 กระจายใกล้เคียงกันทั้ง train/test

# Scaling ข้อมูล (Normalize)
scaler = StandardScaler()  # StandardScaler: ทำให้แต่ละ feature มี mean=0, std=1
X_train_scaled = scaler.fit_transform(X_train)  # fit+transform ข้อมูลฝึก
X_test_scaled = scaler.transform(X_test)  # ใช้ scaler ที่เรียนรู้แล้ว แปลง test

# One-Hot Encoding สำหรับ ANN (บท 10 Neural Network)
encoder = OneHotEncoder(sparse_output=False)  # OneHot = เปลี่ยน class 1,2 → [1,0], [0,1]
y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))  # แปลง y_train
y_test_onehot = encoder.transform(y_test.reshape(-1, 1))  # แปลง y_test

# ----------------------------
# 1–2. Representation & Problem Solving (บท 2)
# ----------------------------
def create_adjacency_matrix(X, k=3):  
    # ฟังก์ชันนี้สร้าง Adjacency Matrix (กราฟแสดงความใกล้กันของข้อมูล)
    n_samples = X.shape[0]  # จำนวนแถวใน X = จำนวน sample
    adj_matrix = np.zeros((n_samples, n_samples))  # สร้าง matrix ศูนย์ (n x n)
    for i in range(n_samples):  # loop ทุก sample
        distances = np.sum((X - X[i])**2, axis=1)  # คำนวณระยะ Euclidean (square) ระหว่าง sample i กับทุกตัว
        nearest_indices = np.argsort(distances)[1:k+1]  # หาค่า index ที่ใกล้ที่สุด k ตัว (ไม่เอาตัวเอง)
        adj_matrix[i, nearest_indices] = 1  # กำหนดค่า 1 ที่ตำแหน่งเพื่อนบ้าน
    return adj_matrix  # คืนค่า adjacency matrix

adj_matrix = create_adjacency_matrix(X_train_scaled)  # สร้าง adjacency matrix ของ training set
adj_html = "<h3>การแทนปัญหาและการแก้ปัญหา (Adjacency Matrix)</h3>"  # HTML สำหรับโชว์ผล

# ----------------------------
# 8. Feature Selection (บท 8)
# ----------------------------
selector = SelectKBest(score_func=f_classif, k=2)  # เลือก feature ที่สำคัญที่สุด 2 ตัว จากทั้งหมด 3
X_train_selected = selector.fit_transform(X_train_scaled, y_train)  # fit+transform ข้อมูลฝึก
X_test_selected = selector.transform(X_test_scaled)  # transform ข้อมูลทดสอบ

# ----------------------------
# 5. Naive Bayes (บท 5)
# ----------------------------
nb = GaussianNB()  # สร้างโมเดล Naive Bayes
nb.fit(X_train_selected, y_train)  # ฝึกโมเดลด้วยข้อมูล train
nb_pred = nb.predict(X_test_selected)  # ทำนาย test set
nb_metrics = compute_metrics(y_test, nb_pred)  # วัดผลด้วย Accuracy, Precision, Recall, F1

# ----------------------------
# 4. Decision Tree (บท 4)
# ----------------------------
dt = DecisionTreeClassifier(random_state=42)  # สร้างโมเดล Decision Tree
dt.fit(X_train_selected, y_train)  # ฝึก
dt_pred = dt.predict(X_test_selected)  # ทำนาย
dt_metrics = compute_metrics(y_test, dt_pred)  # วัดผล

# ----------------------------
# 3 & 6. KNN (บท 3 + 6)
# ----------------------------
knn = KNeighborsClassifier(n_neighbors=5)  # ใช้ K=5 (หาเพื่อนบ้านใกล้สุด 5 ตัว)
knn.fit(X_train_selected, y_train)  # ฝึก
knn_pred = knn.predict(X_test_selected)  # ทำนาย
knn_metrics = compute_metrics(y_test, knn_pred)  # วัดผล

# ----------------------------
# 9. K-Means Clustering (บท 9)
# ----------------------------
kmeans = KMeans(n_clusters=2, random_state=42)  # แบ่งกลุ่มเป็น 2 cluster
kmeans.fit(X_train_selected)  # ฝึก KMeans ด้วย training set
cluster_labels = kmeans.predict(X_test_selected)  # จัดกลุ่ม test set
cluster_score = adjusted_rand_score(y_test, cluster_labels)  # วัดผลการ clustering ด้วย ARI
cluster_html = f"<h3>Clustering (K-Means)</h3><p>Adjusted Rand Score: {cluster_score:.4f}</p>"

# ----------------------------
# 10. ANN (บท 10 Neural Network)
# ----------------------------
ann = tf.keras.Sequential([  # สร้างโมเดลแบบ Sequential
    tf.keras.layers.Input(shape=(2,)),  # input = 2 feature (จาก Feature Selection)
    tf.keras.layers.Dense(10, activation="relu"),  # hidden layer มี 10 นิวรอน, ใช้ ReLU
    tf.keras.layers.Dense(2, activation="softmax")  # output layer มี 2 นิวรอน (class 1/2), ใช้ softmax
])
ann.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])  # กำหนด loss + optimizer
ann.fit(X_train_selected, y_train_onehot, epochs=100, batch_size=5, verbose=0)  # ฝึก ANN 100 รอบ (batch = 5)

ann_pred_onehot = ann.predict(X_test_selected)  # ให้โมเดลทำนาย (output = ความน่าจะเป็น)
ann_pred = np.argmax(ann_pred_onehot, axis=1) + 1  # แปลง one-hot → class (บวก 1 ให้ตรงกับ label 1/2)
ann_metrics = compute_metrics(y_test, ann_pred)  # วัดผล

# ----------------------------
# 7. Cross Validation (บท 7)
# ----------------------------
cv_scores = cross_val_score(DecisionTreeClassifier(random_state=42), X_train_selected, y_train, cv=5)  
# Cross-validation 5 fold (Decision Tree)
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
html_output += print_metrics_html("Naive Bayes", nb_metrics)  # ตารางผล Naive Bayes
html_output += print_metrics_html("Decision Tree", dt_metrics)  # ตารางผล Decision Tree
html_output += print_metrics_html("KNN", knn_metrics)  # ตารางผล KNN
html_output += print_metrics_html("ANN", ann_metrics)  # ตารางผล ANN

html_output += adj_html  # โชว์ adjacency matrix
html_output += cluster_html  # โชว์ผล K-Means
html_output += cv_html  # โชว์ผล Cross-validation
html_output += "</div></body></html>"

with open("haberman_results.html", "w", encoding="utf-8") as f:  
    # บันทึกผลทั้งหมดเป็นไฟล์ HTML
    f.write(html_output)

print("✅ บันทึกผลลัพธ์ HTML ไปที่ 'haberman_results.html' เปิดดูในเบราว์เซอร์ได้เลย")
