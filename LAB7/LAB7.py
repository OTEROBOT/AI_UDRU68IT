import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

# โหลดข้อมูลจากไฟล์ CSV (สมมติว่ามี header หรือไม่มี สามารถปรับได้)
# ในกรณีนี้ haberman.csv ไม่มี header ดังนั้นกำหนด columns
data = pd.read_csv('haberman.csv', header=None)
data.columns = ['age', 'year', 'nodes', 'survival']

# เลือก features ทั้งหมดยกเว้น label (ถ้ามี) เพื่อให้ general สำหรับ dataset ที่มีหลาย features
# ในที่นี้ใช้ age, year, nodes (3 features)
features = data[['age', 'year', 'nodes']].values  # .values เพื่อแปลงเป็น numpy array

# กำหนดจำนวนกลุ่ม (k) สามารถปรับได้ตามต้องการ เช่นใช้ elbow method แต่ที่นี่ใช้ k=2 ตามจำนวน class ใน dataset
k = 2

# ใช้ K-Means Clustering
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(features)

# ผลลัพธ์
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# แสดงผลลัพธ์
print("ผลลัพธ์การแบ่งกลุ่ม (Cluster Labels):")
print(labels)

print("\nตำแหน่งของ Centroids:")
print(centroids)

# สร้างกราฟแสดงการแบ่งกลุ่ม (投影ลง 2D โดยใช้ age และ nodes เพื่อ visualization)
plt.figure(figsize=(8, 6))
plt.scatter(features[:, 0], features[:, 2], c=labels, cmap='rainbow')
plt.scatter(centroids[:, 0], centroids[:, 2], c='black', marker='x', s=100, label='Centroids')
plt.title('การแบ่งกลุ่มด้วย K-Means Clustering บน Haberman Dataset (Age vs Nodes)')
plt.xlabel('Age')
plt.ylabel('Number of Positive Nodes')
plt.legend()
plt.show()

# ถ้าต้องการ evaluate กับ true labels (survival) สามารถทำได้แต่เนื่องจากเป็น unsupervised จึงไม่จำเป็น
# แต่เพื่อตัวอย่าง:
true_labels = data['survival'].values - 1  # แปลง 1,2 เป็น 0,1
from sklearn.metrics import accuracy_score
print("\nAccuracy (โดย match กับ true labels, แต่ไม่ใช่ metric หลักสำหรับ clustering):")
print(accuracy_score(true_labels, labels))  # อาจต่ำเพราะ clustering ไม่รู้ label