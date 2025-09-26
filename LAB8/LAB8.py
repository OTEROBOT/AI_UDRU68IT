import numpy as np  # นำเข้า NumPy สำหรับการคำนวณเชิงตัวเลขและการจัดการอาร์เรย์
import tensorflow as tf  # นำเข้า TensorFlow สำหรับการสร้างและฝึกโมเดล ANN
from sklearn import datasets  # นำเข้าโมดูล datasets จาก scikit-learn เพื่อโหลดชุดข้อมูล Iris
from sklearn.model_selection import train_test_split, cross_val_score  # นำเข้ากระบวนการแบ่งข้อมูลและการตรวจสอบแบบ Cross-Validation
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # นำเข้าการเข้ารหัสและการปรับมาตราส่วนข้อมูล
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, adjusted_rand_score  # นำเข้าการวัดประสิทธิภาพโมเดล
from sklearn.naive_bayes import GaussianNB  # นำเข้า Gaussian Naive Bayes Classifier
from sklearn.tree import DecisionTreeClassifier  # นำเข้า Decision Tree Classifier
from sklearn.neighbors import KNeighborsClassifier  # นำเข้า K-Nearest Neighbors Classifier
from sklearn.cluster import KMeans  # นำเข้า K-Means สำหรับการจัดกลุ่ม
from sklearn.feature_selection import SelectKBest, f_classif  # นำเข้าเครื่องมือคัดเลือกคุณลักษณะ
import pandas as pd  # นำเข้า Pandas สำหรับสร้าง DataFrame และตาราง HTML

# ตั้งค่า seed แบบสุ่มเพื่อให้ผลลัพธ์สม่ำเสมอทุกครั้งที่รัน
np.random.seed(42)  # ตั้งค่า seed สำหรับ NumPy
tf.random.set_seed(42)  # ตั้งค่า seed สำหรับ TensorFlow

# ฟังก์ชันคำนวณเมตริกประเมินผลและส่งคืนเป็นพจนานุกรม
def compute_metrics(y_true, y_pred):  # กำหนดฟังก์ชันโดยรับค่าแท้และค่าที่ทำนาย
    acc = accuracy_score(y_true, y_pred)  # คำนวณความแม่นยำ
    prec = precision_score(y_true, y_pred, average='macro')  # คำนวณความแม่นยำแบบ macro-averaged
    rec = recall_score(y_true, y_pred, average='macro')  # คำนวณอัตราการเรียกคืนแบบ macro-averaged
    f1 = f1_score(y_true, y_pred, average='macro')  # คำนวณคะแนน F1 แบบ macro-averaged
    return {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1}  # ส่งคืนเมตริกเป็นพจนานุกรม

# ฟังก์ชันสร้างตาราง HTML สำหรับแสดงเมตริก
def print_metrics_html(method_name, metrics):  # กำหนดฟังก์ชันรับชื่อวิธีและพจนานุกรมเมตริก
    df = pd.DataFrame([metrics], index=[method_name])  # สร้าง DataFrame จากเมตริกโดยใช้ชื่อวิธีเป็นดัชนี
    html = df.to_html(float_format='%.4f', classes='table table-striped table-bordered table-hover', justify='center', border=1)  # แปลงเป็น HTML พร้อมสไตล์
    return html  # ส่งคืนสตริง HTML

# 1. โหลดชุดข้อมูล Iris - ชุดข้อมูลมาตรฐานที่มี 150 ตัวอย่าง 4 คุณลักษณะ และ 3 คลาส
iris = datasets.load_iris()  # โหลดวัตถุชุดข้อมูล Iris
X = iris.data  # ดึงเมทริกซ์คุณลักษณะ (การวัดกลีบและใบเลี้ยง)
y = iris.target  # ดึงฉลากเป้าหมาย (0, 1, 2 สำหรับสามสปีชีส์)

# 2. แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ (70% ฝึก, 30% ทดสอบ) - รับประกันการประเมินบนข้อมูลที่ไม่เคยเห็น
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # ทำการแบ่งข้อมูลด้วย seed ตายตัว

# 3. ปรับมาตรฐานคุณลักษณะ - ปรับข้อมูลให้มีค่าเฉลี่ย 0 และความแปรปรวน 1 เพื่อเพิ่มประสิทธิภาพโมเดล
scaler = StandardScaler()  # เริ่มต้น StandardScaler
X_train_scaled = scaler.fit_transform(X_train)  # ฝึก scaler กับข้อมูลฝึกและแปลง
X_test_scaled = scaler.transform(X_test)  # แปลงข้อมูลทดสอบด้วย scaler ที่ฝึกแล้ว

# 4. การเข้ารหัสแบบ One-Hot สำหรับ ANN - แปลงฉลากจำนวนเต็มเป็นเวกเตอร์ไบนารีสำหรับการจำแนกหลายคลาสใน ANN
encoder = OneHotEncoder(sparse_output=False)  # เริ่มต้น OneHotEncoder โดยไม่ใช้ output แบบ sparse
y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))  # ฝึกและแปลงฉลากฝึก
y_test_onehot = encoder.transform(y_test.reshape(-1, 1))  # แปลงฉลากทดสอบ

# --- การแทนปัญหาและการแก้ปัญหา (บทที่ 2) ---
# การแสดงโหนดแบบง่ายโดยใช้เมทริกซ์ adjacency - จำลองข้อมูลเป็นกราฟสำหรับการแทนปัญหา
def create_adjacency_matrix(X, k=3):  # กำหนดฟังก์ชันสร้างเมทริกซ์ adjacency ของกราฟ
    n_samples = X.shape[0]  # รับจำนวนตัวอย่าง
    adj_matrix = np.zeros((n_samples, n_samples))  # เริ่มต้นเมทริกซ์ศูนย์ขนาดจัตุรัส
    for i in range(n_samples):  # ลูปผ่านตัวอย่างแต่ละตัว
        distances = np.sum((X - X[i])**2, axis=1)  # คำนวณระยะห่าง Euclidean สี่เหลี่ยมกับทุกตัวอย่าง
        nearest_indices = np.argsort(distances)[1:k+1]  # รับดัชนีของ k ตัวที่ใกล้ที่สุด (ไม่รวมตัวเอง)
        adj_matrix[i, nearest_indices] = 1  # ตั้งค่าเส้นเชื่อมเป็น 1
    return adj_matrix  # ส่งคืนเมทริกซ์
adj_matrix = create_adjacency_matrix(X_train_scaled)  # สร้างเมทริกซ์สำหรับข้อมูลฝึกที่ปรับสเกลแล้ว
adj_html = "<h3>การแทนปัญหาและการแก้ปัญหา (สร้าง Adjacency Matrix)</h3>"  # HTML สำหรับแสดง

# --- การคัดเลือกคุณลักษณะ (บทที่ 8) ---
# เลือก 3 คุณลักษณะชั้นนำโดยใช้ ANOVA F-test - ลดคุณลักษณะให้เหลือแค่ข้อมูลที่มีความสำคัญ
selector = SelectKBest(score_func=f_classif, k=3)  # เริ่มต้น selector ด้วยคะแนน F-classif และ k=3
X_train_selected = selector.fit_transform(X_train_scaled, y_train)  # ฝึกและแปลงข้อมูลฝึก
X_test_selected = selector.transform(X_test_scaled)  # แปลงข้อมูลทดสอบ

# --- Naive Bayes (บทที่ 5) ---
# Gaussian Naive Bayes - ตัวจำแนกเชิงความน่าจะเป็นที่สมมติฐานการกระจายปกติ
nb_model = GaussianNB()  # เริ่มต้นโมเดล
nb_model.fit(X_train_selected, y_train)  # ฝึกด้วยคุณลักษณะที่เลือก
nb_pred = nb_model.predict(X_test_selected)  # ทำนายบนชุดทดสอบ
nb_metrics = compute_metrics(y_test, nb_pred)  # คำนวณเมตริก

# --- Decision Tree (บทที่ 4) ---
# Decision Tree Classifier - สร้างต้นไม้ตามการแบ่งคุณลักษณะ
dt_model = DecisionTreeClassifier(random_state=42)  # เริ่มต้นด้วย seed ตายตัว
dt_model.fit(X_train_selected, y_train)  # ฝึก
dt_pred = dt_model.predict(X_test_selected)  # ทำนาย
dt_metrics = compute_metrics(y_test, dt_pred)  # คำนวณเมตริก

# --- KNN (Nonparametric, บทที่ 6 & การจดจำรูปแบบ, บทที่ 3) ---
# K-Nearest Neighbors - การเรียนรู้ตามตัวอย่างสำหรับการจำแนก
knn_model = KNeighborsClassifier(n_neighbors=5)  # เริ่มต้นด้วย 5 เพื่อนบ้าน
knn_model.fit(X_train_selected, y_train)  # ฝึก
knn_pred = knn_model.predict(X_test_selected)  # ทำนาย
knn_metrics = compute_metrics(y_test, knn_pred)  # คำนวณเมตริก

# --- Clustering (บทที่ 9) ---
# K-Means clustering - แบ่งข้อมูลเป็น 3 กลุ่ม
kmeans = KMeans(n_clusters=3, random_state=42)  # เริ่มต้นด้วย 3 กลุ่ม
kmeans.fit(X_train_selected)  # ฝึกกับข้อมูลฝึก
cluster_labels = kmeans.predict(X_test_selected)  # กำหนดกลุ่มให้ข้อมูลทดสอบ
cluster_score = adjusted_rand_score(y_test, cluster_labels)  # คำนวณคะแนน ARI สำหรับการประเมิน
cluster_html = f"<h3>Clustering (K-Means)</h3><p>Adjusted Rand Score: {cluster_score:.4f}</p>"  # HTML สำหรับผลลัพธ์การจัดกลุ่ม

# --- ANN (บทที่ 10) ---
# Artificial Neural Network - Perceptron หลายชั้นสำหรับการจำแนก
ann_model = tf.keras.Sequential([  # สร้างโมเดลแบบต่อเนื่อง
    tf.keras.layers.Input(shape=(3,)),  # ชั้นนำเข้าตรงกับคุณลักษณะที่เลือก
    tf.keras.layers.Dense(10, activation='relu'),  # ชั้นซ่อนหนาแน่นด้วย ReLU
    tf.keras.layers.Dense(3, activation='softmax')  # ชั้นเอาต์พุตด้วย softmax สำหรับความน่าจะเป็น
])
ann_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # คอมไพล์โมเดล
ann_model.fit(X_train_selected, y_train_onehot, epochs=200, batch_size=5, verbose=0)  # ฝึกโมเดลแบบเงียบ
ann_pred_onehot = ann_model.predict(X_test_selected)  # รับการทำนายความน่าจะเป็น
ann_pred = np.argmax(ann_pred_onehot, axis=1)  # แปลงความน่าจะเป็นเป็นฉลากคลาส
ann_metrics = compute_metrics(y_test, ann_pred)  # คำนวณเมตริก

# --- การทดสอบโมเดล (บทที่ 7) - Cross-Validation ---
# ประเมินความแข็งแกร่งของโมเดลโดยใช้ Cross-Validation 5 เท่าใน Decision Tree
cv_scores = cross_val_score(DecisionTreeClassifier(random_state=42), X_train_selected, y_train, cv=5)  # รัน CV
cv_mean = cv_scores.mean()  # คำนวณคะแนนเฉลี่ย
cv_std = cv_scores.std() * 2  # คำนวณช่วงความเชื่อมั่น 95%
cv_html = f"<h3>การทดสอบโมเดล (Cross-Validation บน Decision Tree)</h3><p>Mean CV Score: {cv_mean:.4f} (+/- {cv_std:.4f})</p>"  # HTML สำหรับผลลัพธ์ CV

# สร้างผลลัพธ์ HTML เต็มรูปแบบสำหรับแสดงผลบนเว็บอย่างสวยงาม
html_output = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ผลลัพธ์การเปรียบเทียบวิธี AI</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">  <!-- Bootstrap สำหรับสไตล์ -->
    <style>
        body { padding: 20px; background-color: #f8f9fa; }
        h1, h2, h3 { color: #343a40; }
        .table { margin: auto; width: 80%; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center">ผลลัพธ์การเปรียบเทียบวิธี AI</h1>
        <h2>วิธีการจำแนก</h2>
"""  # เริ่มโครงสร้าง HTML ด้วย Bootstrap สำหรับการออกแบบตอบสนอง

# เพิ่มตารางการจำแนก
html_output += print_metrics_html("Naive Bayes", nb_metrics)  # เพิ่มตาราง Naive Bayes
html_output += print_metrics_html("Decision Tree", dt_metrics)  # เพิ่มตาราง Decision Tree
html_output += print_metrics_html("KNN", knn_metrics)  # เพิ่มตาราง KNN
html_output += print_metrics_html("ANN", ann_metrics)  # เพิ่มตาราง ANN

# เพิ่มส่วนอื่นๆ
html_output += adj_html  # เพิ่มส่วนการแทนปัญหา
html_output += cluster_html  # เพิ่มส่วนการจัดกลุ่ม
html_output += cv_html  # เพิ่มส่วนการทดสอบโมเดล

html_output += """
    </div>
</body>
</html>
"""  # ปิด HTML

# บันทึก HTML ลงไฟล์เพื่อดูผลลัพธ์บนเว็บได้ง่าย (เปิดในเบราว์เซอร์)
with open('ai_results.html', 'w', encoding='utf-8') as f:  # เปิดไฟล์ในโหมดเขียนด้วย UTF-8
    f.write(html_output)  # เขียนเนื้อหา HTML

print("บันทึกผลลัพธ์ HTML ไปที่ 'ai_results.html' เปิดในเบราว์เซอร์เพื่อดูการแสดงผลที่สวยงาม")  # ข้อความในคอนโซล