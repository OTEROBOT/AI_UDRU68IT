# คู่มือการใช้งานโค้ด LAB7: K-Means Clustering (Haberman Dataset)

## 📌 สิ่งที่โค้ดนี้ทำ
- โหลดข้อมูลจากไฟล์ `haberman.csv`
- ทำการจัดกลุ่ม (Clustering) ด้วย **K-Means**
- แสดงผลลัพธ์การจัดกลุ่ม + centroid
- วาดกราฟ (Age vs Nodes)
- (เพิ่มเติม) แสดง Accuracy เมื่อเทียบกับ label จริง

---

## ⚙️ สิ่งที่ต้องติดตั้งก่อนใช้งาน
ต้องใช้ **Python 3.8+** และไลบรารีดังนี้:


pip install numpy pandas matplotlib scikit-learn
```

---

## ▶️ วิธีใช้งาน
1. ดาวน์โหลดไฟล์โค้ด (เช่น `LAB7.py`) และไฟล์ `haberman.csv` มาไว้ในโฟลเดอร์เดียวกัน  
2. เปิด Terminal/Command Prompt ที่โฟลเดอร์นั้น  
3. รันคำสั่ง:
   
   python LAB7.py
   ```
4. ผลลัพธ์ที่ได้:
   - Cluster Labels
   - ตำแหน่ง Centroids
   - กราฟแสดงการแบ่งกลุ่ม (Age vs Nodes)
   - ค่า Accuracy (อ้างอิงเทียบกับ survival label)

---

## 📝 หมายเหตุ
- ไฟล์ `haberman.csv` ต้องอยู่ในโฟลเดอร์เดียวกับโค้ด
- ถ้า CSV มี header ต้องแก้ `header=None` ในโค้ด
- ค่า `k` (จำนวน cluster) สามารถเปลี่ยนได้ตาม dataset
