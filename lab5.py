import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# ========== โหลด Dataset ==========
filename = r"D:\armandostudy\codeai\healthcaredatasetstrokedata.csv"
df = pd.read_csv(filename)
df.drop(columns=['id'], inplace=True)

# เลือก Features และ Labels
features = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
X = df[features].fillna(df[features].mean()).values
y = df["stroke"].values  # label (0 = ไม่เป็น, 1 = เป็น)

# ========== Models ==========
models = {
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3)
}

# ========== Cross Validation ==========
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# สร้าง list สำหรับเก็บค่าเฉลี่ยของทุกโมเดล
all_model_averages = []

for model_name, model in models.items():
    print("=" * 50)
    print(f"MODEL: {model_name}")
    print("=" * 50)

    # สร้าง DataFrame สำหรับเก็บผลลัพธ์ของแต่ละ fold
    results = []

    fold = 1
    for train_index, test_index in kf.split(X):
        # Split train/test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Confusion Matrix
        labels = [0, 1]  # 0 = ไม่เป็น, 1 = เป็น
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        # Metrics (%)
        accuracy = accuracy_score(y_test, y_pred) * 100
        precision = precision_score(y_test, y_pred, average="binary", zero_division=0) * 100
        recall = recall_score(y_test, y_pred, average="binary", zero_division=0) * 100
        f1 = f1_score(y_test, y_pred, average="binary", zero_division=0) * 100

        # เก็บผลลัพธ์ใน list
        results.append({
            "Fold": fold,
            "Accuracy": accuracy,
            "Recall": recall,
            "Precision": precision,
            "F-Measure": f1
        })

        # Show results
        print(f"\n--- Fold {fold} ---")
        print("Confusion Matrix:")
        print(pd.DataFrame(cm,
                           index=[f"Actual {l}" for l in labels],
                           columns=[f"Pred {l}" for l in labels]))
        print(f"Accuracy : {accuracy:.2f}%")
        print(f"Recall   : {recall:.2f}%")
        print(f"Precision: {precision:.2f}%")
        print(f"F-Measure: {f1:.2f}%")
        print("-" * 40)
        fold += 1

    # สร้างตารางสรุปผลลัพธ์ของแต่ละโมเดล
    results_df = pd.DataFrame(results)
    
    # เพิ่มแถวค่าเฉลี่ย
    avg_row = {
        "Fold": "AVG",
        "Accuracy": results_df["Accuracy"].mean(),
        "Recall": results_df["Recall"].mean(),
        "Precision": results_df["Precision"].mean(),
        "F-Measure": results_df["F-Measure"].mean()
    }
    results_df = pd.concat([results_df, pd.DataFrame([avg_row])], ignore_index=True)

    # จัดรูปแบบให้แสดงเป็น %.2f%
    results_df["Accuracy"] = results_df["Accuracy"].map("{:.2f}%".format)
    results_df["Recall"] = results_df["Recall"].map("{:.2f}%".format)
    results_df["Precision"] = results_df["Precision"].map("{:.2f}%".format)
    results_df["F-Measure"] = results_df["F-Measure"].map("{:.2f}%".format)

    print("\nSummary Table for", model_name)
    print("=" * 50)
    print(results_df.to_string(index=False))
    print("\n")  # เว้นบรรทัดหลังจบแต่ละโมเดล

    # เก็บค่าเฉลี่ยของโมเดลนี้สำหรับตารางรวม
    all_model_averages.append({
        "Model": model_name,
        "Accuracy": avg_row["Accuracy"],
        "Recall": avg_row["Recall"],
        "Precision": avg_row["Precision"],
        "F-Measure": avg_row["F-Measure"]
    })

# สร้างตารางสรุปค่าเฉลี่ยของทุกโมเดล
avg_df = pd.DataFrame(all_model_averages)

# เพิ่มแถวค่าเฉลี่ยรวมของทุกโมเดล
overall_avg_row = {
    "Model": "Overall AVG",
    "Accuracy": avg_df["Accuracy"].mean(),
    "Recall": avg_df["Recall"].mean(),
    "Precision": avg_df["Precision"].mean(),
    "F-Measure": avg_df["F-Measure"].mean()
}
avg_df = pd.concat([avg_df, pd.DataFrame([overall_avg_row])], ignore_index=True)

# จัดรูปแบบให้แสดงเป็น %.2f%
avg_df["Accuracy"] = avg_df["Accuracy"].map("{:.2f}%".format)
avg_df["Recall"] = avg_df["Recall"].map("{:.2f}%".format)
avg_df["Precision"] = avg_df["Precision"].map("{:.2f}%".format)
avg_df["F-Measure"] = avg_df["F-Measure"].map("{:.2f}%".format)

print("=" * 50)
print("Summary Table of Average Metrics for All Models")
print("=" * 50)
print(avg_df.to_string(index=False))
print("\n")