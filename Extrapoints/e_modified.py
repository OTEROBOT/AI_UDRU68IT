import numpy as np
from scipy.stats import multivariate_normal
from tabulate import tabulate  # ต้องติดตั้งด้วย pip install tabulate

#Assignmentแก้ไขครั้งที่ 1.3

# โหลดข้อมูลจากไฟล์ TWOCLASS.dat
data = np.loadtxt('TWOCLASS.dat')
class1 = data[0:100, :]  # แบ่งข้อมูล Class 1 (แถว 1-100)
class2 = data[100:200, :]  # แบ่งข้อมูล Class 2 (แถว 101-200)

# ฟังก์ชันสำหรับคำนวณเวกเตอร์ค่าเฉลี่ย (Mean Vector) - สูตร 1
def calculate_mean(data_subset):
    return np.mean(data_subset, axis=0)

# ฟังก์ชันสำหรับคำนวณเมทริกซ์ความแปรปรวน (Covariance Matrix) - สูตร 2 (ML)
def calculate_covariance(data_subset, mean):
    n_samples = data_subset.shape[0]  # จำนวนตัวอย่างในข้อมูลย่อย
    centered_data = data_subset - mean  # คำนวณความแตกต่างจากค่าเฉลี่ย
    return np.dot(centered_data.T, centered_data) / n_samples  # คำนวณ covariance โดยใช้ Maximum Likelihood

# ฟังก์ชันสำหรับจำแนกคลาสของตัวอย่างโดยใช้กฎ Bayes Decision - สูตร 3 + Decision Rule
def classify(x, mean1, mean2, cov1, cov2):
    dist1 = multivariate_normal(mean=mean1, cov=cov1, allow_singular=True)  # สร้างการแจกแจงปกติสำหรับ Class 1
    dist2 = multivariate_normal(mean=mean2, cov=cov2, allow_singular=True)  # สร้างการแจกแจงปกติสำหรับ Class 2
    p1 = dist1.pdf(x)  # คำนวณความน่าจะเป็นของ Class 1 (สูตร 3)
    p2 = dist2.pdf(x)  # คำนวณความน่าจะเป็นของ Class 2 (สูตร 3)
    if p1 > p2:
        return 1  # กลับค่า 1 หาก Class 1 มีความน่าจะเป็นสูงกว่า (Decision Rule)
    elif p1 < p2:
        return 2  # กลับค่า 2 หาก Class 2 มีความน่าจะเป็นสูงกว่า
    else:
        return np.random.choice([1, 2])  # สุ่มเลือกคลาสหากความน่าจะเป็นเท่ากัน

# ฟังก์ชันหลักสำหรับทดสอบโมเดล (เหมือนเดิม แต่เพิ่ม comment สูตร)
def test_model(class1, class2, features, feature_desc):
    n_features = len(features)  # จำนวนฟีเจอร์ที่ใช้
    feat_names = [f + 1 for f in features]  # แปลงดัชนีฟีเจอร์เป็นหมายเลข (1, 2, 3, ...)
    print(f"\n{'='*40}")
    print(f"การทดสอบกับ {feature_desc} (Features: {', '.join(map(str, feat_names))})")
    print(f"{'='*40}")

    # ดึงข้อมูลฟีเจอร์และเลเบลที่เลือก
    X1 = class1[:, features]  # ข้อมูล Class 1
    X2 = class2[:, features]  # ข้อมูล Class 2
    Y1 = class1[:, 4]  # เลเบล Class 1
    Y2 = class2[:, 4]  # เลเบล Class 2

    # ทำการทดสอบแบบ Cross-Validation 10% และทดสอบเต็มรูปแบบ (11 passes)
    accuracies = []
    for test_id in range(1, 12):
        if test_id < 11:
            start_idx = (test_id - 1) * 10  # ดัชนีเริ่มต้นของชุดทดสอบ
            end_idx = test_id * 10  # ดัชนีสิ้นสุดของชุดทดสอบ
            print(f"\nTest {test_id}: Test the data at {start_idx+1}-{end_idx} From all data")
            train1 = np.vstack((class1[:start_idx], class1[end_idx:]))  # ชุดฝึก Class 1
            train2 = np.vstack((class2[:start_idx], class2[end_idx:]))  # ชุดฝึก Class 2
            test1 = class1[start_idx:end_idx]  # ชุดทดสอบ Class 1
            test2 = class2[start_idx:end_idx]  # ชุดทดสอบ Class 2
        else:
            print(f"\nTest {test_id}: Test the data at 1-100 From all data")
            train1 = class1  # ใช้ข้อมูลทั้งหมดสำหรับฝึก Class 1
            train2 = class2  # ใช้ข้อมูลทั้งหมดสำหรับฝึก Class 2
            test1 = class1  # ใช้ข้อมูลทั้งหมดสำหรับทดสอบ Class 1
            test2 = class2  # ใช้ข้อมูลทั้งหมดสำหรับทดสอบ Class 2

        # คำนวณเวกเตอร์ค่าเฉลี่ย - สูตร 1
        mean1 = calculate_mean(train1[:, features])
        mean2 = calculate_mean(train2[:, features])

        # คำนวณเมทริกซ์ความแปรปรวน - สูตร 2
        cov1 = calculate_covariance(train1[:, features], mean1)
        cov2 = calculate_covariance(train2[:, features], mean2)

        # แสดงเวกเตอร์ค่าเฉลี่ยในตารางที่จัดรูปแบบ
        mean_table = [
            ["Mean w1"] + [f"{x:.4f}" for x in mean1],
            ["Mean w2"] + [f"{x:.4f}" for x in mean2]
        ]
        print("\nMean Vectors:")  # แสดงหัวข้อเวกเตอร์ค่าเฉลี่ย
        print(tabulate(mean_table, headers=[""] + [f"Feature {i+1}" for i in range(n_features)], tablefmt="grid"))

        # แสดงเมทริกซ์ความแปรปรวนในตารางที่จัดรูปแบบ (row-major order)
        cov1_flat = [f"{cov1[i,j]:.4f}" for j in range(n_features) for i in range(n_features)]
        cov2_flat = [f"{cov2[i,j]:.4f}" for j in range(n_features) for i in range(n_features)]
        cov1_table = [["Cov w1"] + cov1_flat]
        cov2_table = [["Cov w2"] + cov2_flat]
        print("\nCovariance Matrices (Row-wise):")  # แสดงหัวข้อเมทริกซ์ความแปรปรวน
        print(tabulate(cov1_table, headers=[""] + [f"({i+1},{j+1})" for j in range(n_features) for i in range(n_features)], tablefmt="grid"))
        print(tabulate(cov2_table, headers=[""] + [f"({i+1},{j+1})" for j in range(n_features) for i in range(n_features)], tablefmt="grid"))

        # จำแนกข้อมูลทดสอบ - สูตร 3 + Decision
        pred1 = [classify(test1[i, features], mean1, mean2, cov1, cov2) for i in range(len(test1))]
        pred2 = [classify(test2[i, features], mean1, mean2, cov1, cov2) for i in range(len(test2))]
        actual_pred = np.array(pred1 + pred2)
        truth = np.concatenate([test1[:, 4], test2[:, 4]])

        # สร้างเมทริกซ์ความสับสน (Confusion Matrix)
        conf = np.zeros((2, 2), dtype=int)
        for p, t in zip(actual_pred, truth):
            conf[int(t) - 1, int(p) - 1] += 1

        # แสดงตาราง Actual Table
        actual_table = [["Actual"]] + [[f"Class {i+1}"] + [f"{conf[i,j]}" for j in range(2)] for i in range(2)]
        print("\nActual Table:")  # แสดงหัวข้อตารางผลลัพธ์
        print(tabulate(actual_table, headers=[""] + ["Class 1", "Class 2"], tablefmt="grid"))

        # คำนวณความถูกต้อง
        acc = np.sum(actual_pred == truth) / len(truth) * 100
        print(f"Accuracy (Correct) = {acc:.2f}%")
        print(" ")
        print(" ")
        print(" ")
        accuracies.append(acc)

    # คำนวณและแสดงค่าเฉลี่ยความถูกต้อง
    avg_acc = np.mean(accuracies)
    print(f"\n>> Average Accuracy: {avg_acc:.2f}%")

# รันการทดสอบ (เหมือนเดิม + เพิ่ม Test 3 สำหรับ demo)
print("1. การทดสอบที่ 1: ทดสอบกับ 4 Features")
test_model(class1, class2, [0, 1, 2, 3], "4 Features")

print("\n2. การทดสอบที่ 2: ทดสอบกับ 2 Features (Feature 1 และ Feature 2)")
test_model(class1, class2, [0, 1], "2 Features (Feature 1 และ Feature 2)")

print("\n3. การทดสอบที่ 3: ทดสอบกับ 2 Features (Feature 2 และ Feature 4) - สำหรับ demo ถ้าจารย์ถาม")
test_model(class1, class2, [1, 3], "2 Features (Feature 2 และ Feature 4)")