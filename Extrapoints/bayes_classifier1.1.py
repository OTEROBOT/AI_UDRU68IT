import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import multivariate_normal

# โหลดข้อมูลจากไฟล์ TWOCLASS.dat เข้าสู่ตัวแปร data
data = np.loadtxt("TWOCLASS.dat")

# แบ่งข้อมูลออกเป็น 2 คลาส:
# class1 = แถวที่ 1-100, class2 = แถวที่ 101-200
class1 = data[0:100, :]
class2 = data[100:200, :]

# ฟังก์ชันสำหรับคำนวณ Covariance Matrix
def covar(x, mean):
    diff = x - mean                       # คำนวณความแตกต่างระหว่างแต่ละจุดกับ mean
    return np.dot(diff.T, diff) / len(x) # คำนวณ covariance แบบ maximum likelihood

# ฟังก์ชันในการจำแนกคลาสของตัวอย่าง x
def classify(x, mean1, mean2, cov1, cov2):
    # สร้าง distribution สำหรับแต่ละคลาส (Gaussian Multivariate)
    dist1 = multivariate_normal(mean=mean1, cov=cov1, allow_singular=True)
    dist2 = multivariate_normal(mean=mean2, cov=cov2, allow_singular=True)

    # คำนวณความน่าจะเป็นของ x ภายใต้แต่ละคลาส
    p1 = dist1.pdf(x)
    p2 = dist2.pdf(x)

    # ตัดสินใจว่าอยู่คลาสไหนตามความน่าจะเป็น
    if p1 > p2:
        return 1
    elif p1 < p2:
        return 2
    else:
        return np.random.choice([1, 2])  # ถ้าเท่ากันสุ่มคลาส

# ฟังก์ชันสำหรับจัดรูปแบบการพิมพ์ตาราง
def print_table(data, headers, title="Actual"):
    print(f"\n{title}")
    print(" " + " ".join(f"{h:^6}" for h in headers))
    for i, row in enumerate(data):
        print(f"{i+1} {' '.join(f'{x:^6.0f}' for x in row)}")

# ฟังก์ชันหลักในการทดสอบโมเดล
def test_model(class1, class2, features):
    n_features = len(features)
    feat_names = [f + 1 for f in features]
    print(f"\n{'='*20} ใช้ features {', '.join(map(str, feat_names))} ({n_features} features) {'='*20}")

    # ดึงเฉพาะ features ที่ต้องการใช้จากข้อมูล class1 และ class2
    X1 = class1[:, features]
    X2 = class2[:, features]

    # ดึง label (class) ซึ่งอยู่ที่ column 5 (index 4)
    Y1 = class1[:, 4]
    Y2 = class2[:, 4]

    # ใช้ K-Fold cross-validation แบ่งข้อมูลเป็น 10 ส่วน (10 fold)
    kf = KFold(n_splits=10, shuffle=False)
    fold = 1
    accuracy_all = []

    # วนลูปการเทรนและทดสอบในแต่ละ fold
    for (train_index, test_index) in kf.split(X1):
        # เตรียมข้อมูลสำหรับฝึก
        x1_train = X1[train_index]
        x2_train = X2[train_index]
        # คำนวณค่าเฉลี่ย (mean vector) สำหรับแต่ละ class
        mean1 = np.mean(x1_train, axis=0)
        mean2 = np.mean(x2_train, axis=0)
        # คำนวณ covariance matrix สำหรับแต่ละ class
        cov1 = covar(x1_train, mean1)
        cov2 = covar(x2_train, mean2)

        print(f"\n--- Fold {fold} ---")
        print(f"Mean Class 1: {np.round(mean1, 4)}")
        print(f"Mean Class 2: {np.round(mean2, 4)}")
        print(f"Cov Class 1:\n{np.round(cov1, 4)}")
        print(f"Cov Class 2:\n{np.round(cov2, 4)}")

        # เตรียมข้อมูลสำหรับทดสอบ
        x1_test = X1[test_index]
        x2_test = X2[test_index]
        y1_true = Y1[test_index]
        y2_true = Y2[test_index]

        # ทำนายคลาสของข้อมูล test
        pred1 = [classify(x, mean1, mean2, cov1, cov2) for x in x1_test]
        pred2 = [classify(x, mean1, mean2, cov1, cov2) for x in x2_test]

        # รวมผลลัพธ์ของการทดสอบทั้งสอง class
        actual_pred = np.array(pred1 + pred2)
        truth = np.concatenate([y1_true, y2_true])

        # Confusion matrix
        conf = np.zeros((2, 2), dtype=int)
        for p, t in zip(actual_pred, truth):
            conf[int(t) - 1, int(p) - 1] += 1

        print_table(conf, ["1", "2"])

        # คำนวณความถูกต้อง (accuracy)
        acc = np.sum(actual_pred == truth) / len(truth) * 100
        accuracy_all.append(acc)
        print(f"Accuracy: {acc:.2f}%")
        fold += 1

    # ทดสอบครั้งสุดท้ายด้วยการใช้ข้อมูลทั้งหมด (train + test = 100%)
    mean1 = np.mean(X1, axis=0)
    mean2 = np.mean(X2, axis=0)
    cov1 = covar(X1, mean1)
    cov2 = covar(X2, mean2)

    pred1 = [classify(x, mean1, mean2, cov1, cov2) for x in X1]
    pred2 = [classify(x, mean1, mean2, cov1, cov2) for x in X2]
    actual_pred = np.array(pred1 + pred2)
    truth = np.concatenate([Y1, Y2])

    print(f"\n--- Test 11 (Full test) ---")
    print(f"Mean Class 1: {np.round(mean1, 4)}")
    print(f"Mean Class 2: {np.round(mean2, 4)}")
    print(f"Cov Class 1:\n{np.round(cov1, 4)}")
    print(f"Cov Class 2:\n{np.round(cov2, 4)}")

    conf = np.zeros((2, 2), dtype=int)
    for p, t in zip(actual_pred, truth):
        conf[int(t) - 1, int(p) - 1] += 1

    print_table(conf, ["1", "2"])

    acc = np.sum(actual_pred == truth) / len(truth) * 100
    accuracy_all.append(acc)
    print(f"Accuracy: {acc:.2f}%")

    # แสดงค่า accuracy เฉลี่ยจากทั้งหมด 11 ครั้ง
    print(f">> Average Accuracy: {np.mean(accuracy_all):.2f}%")

# จุดเริ่มต้นของโปรแกรม
if __name__ == "__main__":
    # ทดสอบแบบใช้ 4 ฟีเจอร์ทั้งหมด (features 1,2,3,4)
    test_model(class1, class2, features=[0,1,2,3])
    # ทดสอบแบบใช้แค่ฟีเจอร์ 1 และ 2
    test_model(class1, class2, features=[0,1])
    # ทดสอบแบบใช้แค่ฟีเจอร์ 1 และ 4
    test_model(class1, class2, features=[0,3])