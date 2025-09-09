import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# ฟังก์ชันสำหรับพิมพ์ Confusion Matrix ให้สวยงามแบบ ASCII table
def print_pretty_cm(cm, fold, method):
    # พิมพ์กรอบบนของตาราง
    print(f"╔════════════════════════════════════════╗")
    # พิมพ์หัวตารางระบุว่าเป็น Method และ Fold ใด
    print(f"║ {method} Fold {fold} Confusion Matrix  ║")
    # พิมพ์เส้นแบ่งระหว่างหัวตาราง
    print(f"╠════════════╦═════════════╦═════════════╣")
    # พิมพ์หัวคอลัมน์ Predicted
    print(f"║            ║ Predicted 1 ║ Predicted 2 ║")
    # พิมพ์เส้นแบ่งระหว่างแถว
    print(f"╠════════════╬═════════════╬═════════════╣")
    # พิมพ์แถว Actual 1 พร้อมค่า Confusion Matrix
    print(f"║ Actual 1   ║     {cm[0,0]:<5}   ║     {cm[0,1]:<5}   ║")
    # พิมพ์เส้นแบ่งระหว่างแถว
    print(f"╠════════════╬═════════════╬═════════════╣")
    # พิมพ์แถว Actual 2 พร้อมค่า Confusion Matrix
    print(f"║ Actual 2   ║     {cm[1,0]:<5}   ║     {cm[1,1]:<5}   ║")
    # พิมพ์กรอบล่างของตาราง
    print(f"╚════════════╩═════════════╩═════════════╝")

# ฟังก์ชันสำหรับพิมพ์ค่า Metrics
def print_pretty_metrics(accuracy, recall, precision, f_measure):
    # พิมพ์ค่า Accuracy
    print(f"║ Accuracy:   {accuracy * 100:6.2f}%                    ║")
    # พิมพ์ค่า Recall
    print(f"║ Recall:     {recall * 100:6.2f}%                    ║")
    # พิมพ์ค่า Precision
    print(f"║ Precision:  {precision * 100:6.2f}%                    ║")
    # พิมพ์ค่า F-Measure
    print(f"║ F-Measure:  {f_measure * 100:6.2f}%                    ║")
    # พิมพ์กรอบล่างของตาราง Metrics
    print(f"╚════════════════════════════════════════╝\n")

# ฟังก์ชัน K-Nearest Neighbor
def k_nearest_neighbor(X_train, X_test, y_train):
    # จำนวนข้อมูลฝึก
    n = len(X_train)
    # กำหนด k ตามสูตร k = sqrt(n)
    k = int(np.sqrt(n))
    # สร้างลิสต์สำหรับเก็บการทำนาย
    predictions = []
    
    # ลูปสำหรับแต่ละจุดทดสอบ
    for x_test in X_test:
        # คำนวณระยะห่าง Euclidean ระหว่างจุดทดสอบและจุดฝึก
        distances = np.sqrt(np.sum((X_train - x_test) ** 2, axis=1))
        # หา k ค่าใกล้ที่สุด
        nearest_indices = np.argsort(distances)[:k]
        # รับ Label ของ k เพื่อนบ้านที่ใกล้ที่สุด
        nearest_labels = y_train[nearest_indices]
        # ทำนาย Class ด้วยการนับจำนวนที่มากที่สุด
        pred = np.bincount(nearest_labels.astype(int), minlength=3)[1:].argmax() + 1
        # เพิ่มการทำนายไปยังลิสต์
        predictions.append(pred)
    # คืนค่าการทำนายเป็น NumPy array
    return np.array(predictions)

# โหลด dataset
df = pd.read_csv('haberman.csv', header=None)
# กำหนดชื่อคอลัมน์ให้ชัดเจน
df.columns = ['age', 'year_of_operation', 'positive_axillary_nodes', 'survival_status']
# แปลงข้อมูลเป็น NumPy array สำหรับ X และ y
X = df[['age', 'year_of_operation', 'positive_axillary_nodes']].values
y = df['survival_status'].values

# ตั้งค่า Cross-validation 5-fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# เตรียมเก็บ Metrics สำหรับแต่ละ Model
models = {
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3)
}

# Dictionary เพื่อเก็บค่าเฉลี่ย Metrics ของแต่ละโมเดล
model_metrics = {}

for model_name, model in models.items():
    accuracies = []  # เก็บค่า Accuracy
    recalls = []     # เก็บค่า Recall
    precisions = []  # เก็บค่า Precision
    f_measures = []  # เก็บค่า F-Measure

    fold = 1  # ตัวนับ Fold
    # ลูปสำหรับแต่ละ Fold
    for train_index, test_index in kf.split(X):
        # แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # ทำนายด้วย Model ที่เลือก
        if model_name == "KNN (k=3)":
            y_pred = k_nearest_neighbor(X_train, X_test, y_train)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # สร้าง Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=[1, 2])
        print_pretty_cm(cm, fold, model_name)
        
        # คำนวณ Metrics
        TP = cm[0, 0]  # True Positive
        FN = cm[0, 1]  # False Negative
        FP = cm[1, 0]  # False Positive
        TN = cm[1, 1]  # True Negative
        total = TP + TN + FP + FN
        accuracy = (TP + TN) / total if total != 0 else 0  # คำนวณ Accuracy
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0  # คำนวณ Precision
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0  # คำนวณ Recall
        f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0  # คำนวณ F-Measure
        
        print(f"╔════════════════════════════════════════╗")
        print_pretty_metrics(accuracy, recall, precision, f_measure)
        
        # เก็บค่า Metrics ไว้คำนวณเฉลี่ย
        accuracies.append(accuracy)
        recalls.append(recall)
        precisions.append(precision)
        f_measures.append(f_measure)
        
        fold += 1  # ไป Fold ถัดไป

    # คำนวณค่าเฉลี่ยของ Metrics
    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_recall = sum(recalls) / len(recalls)
    avg_precision = sum(precisions) / len(precisions)
    avg_f_measure = sum(f_measures) / len(f_measures)

    # เก็บค่าเฉลี่ยใน dictionary
    model_metrics[model_name] = {
        "Accuracy": avg_accuracy,
        "Recall": avg_recall,
        "Precision": avg_precision,
        "F-Measure": avg_f_measure
    }

    print(f"╔════════════════════════════════════════╗")
    print(f"║       Average {model_name} Metrics         ║")
    print(f"╠════════════════════════════════════════╣")
    print_pretty_metrics(avg_accuracy, avg_recall, avg_precision, avg_f_measure)

# พิมพ์ตารางสรุปของโมเดลทั้ง 3
print(f"╔════════════════════════════════════════════════════════════════════╗")
print(f"║                  Summary of Model Performance                     ║")
print(f"╠════════════╦════════════╦════════════╦═════════════╦═════════════╣")
print(f"║ Model      ║ Accuracy   ║ Recall     ║ Precision   ║ F-Measure   ║")
print(f"╠════════════╬════════════╬════════════╬═════════════╬═════════════╣")
for model_name, metrics in model_metrics.items():
    print(f"║ {model_name:<10} ║ {metrics['Accuracy']*100:>6.2f}% ║ {metrics['Recall']*100:>6.2f}% ║ {metrics['Precision']*100:>6.2f}% ║ {metrics['F-Measure']*100:>6.2f}% ║")
print(f"╚════════════╩════════════╩════════════╩═════════════╩═════════════╝")