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

# ฟังก์ชันหลักในการทดสอบโมเดล
def test_model(class1, class2, n_features):
    print(f"\n===== ใช้ {n_features} features =====")

    # ดึงเฉพาะ features ที่ต้องการใช้จากข้อมูล class1 และ class2
    X1 = class1[:, :n_features]
    X2 = class2[:, :n_features]

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
        x1_train, x2_train = X1[train_index], X2[train_index]
        # คำนวณค่าเฉลี่ย (mean vector) สำหรับแต่ละ class
        mean1 = np.mean(x1_train, axis=0)
        mean2 = np.mean(x2_train, axis=0)
        # คำนวณ covariance matrix สำหรับแต่ละ class
        cov1 = covar(x1_train, mean1)
        cov2 = covar(x2_train, mean2)

        # เตรียมข้อมูลสำหรับทดสอบ
        x1_test, x2_test = X1[test_index], X2[test_index]
        y1_true, y2_true = Y1[test_index], Y2[test_index]

        # ทำนายคลาสของข้อมูล test
        pred1 = [classify(x, mean1, mean2, cov1, cov2) for x in x1_test]
        pred2 = [classify(x, mean1, mean2, cov1, cov2) for x in x2_test]

        # รวมผลลัพธ์ของการทดสอบทั้งสอง class
        actual = np.array(pred1 + pred2)
        truth = np.concatenate([y1_true, y2_true])

        # คำนวณความถูกต้อง (accuracy)
        acc = np.sum(actual == truth) / len(truth) * 100
        accuracy_all.append(acc)
        print(f"Test {fold} - Accuracy: {acc:.2f}%")
        fold += 1

    # ทดสอบครั้งสุดท้ายด้วยการใช้ข้อมูลทั้งหมด (train + test = 100%)
    mean1 = np.mean(X1, axis=0)
    mean2 = np.mean(X2, axis=0)
    cov1 = covar(X1, mean1)
    cov2 = covar(X2, mean2)

    pred1 = [classify(x, mean1, mean2, cov1, cov2) for x in X1]
    pred2 = [classify(x, mean1, mean2, cov1, cov2) for x in X2]
    actual = np.array(pred1 + pred2)
    truth = np.concatenate([Y1, Y2])
    acc = np.sum(actual == truth) / len(truth) * 100
    accuracy_all.append(acc)
    print(f"Test 11 (Full test) - Accuracy: {acc:.2f}%")

    # แสดงค่า accuracy เฉลี่ยจากทั้งหมด 11 ครั้ง
    print(f">> Average Accuracy ({n_features} features): {np.mean(accuracy_all):.2f}%")

# จุดเริ่มต้นของโปรแกรม
if __name__ == "__main__":
    # ทดสอบแบบใช้ 4 ฟีเจอร์ทั้งหมด
    test_model(class1, class2, n_features=4)
    # ทดสอบแบบใช้แค่ฟีเจอร์ 1 และ 2
    test_model(class1, class2, n_features=2)
