import numpy as np
from scipy.stats import multivariate_normal
from tabulate import tabulate  # pip install tabulate ถ้ายังไม่มี
import sys

#Assignment แก้ไขครั้งที่ 1.4

# Generate synthetic data to match PDF means (uncomment load if you have file)
# data = np.loadtxt('TWOCLASS.dat')
# class1 = data[0:100, :]
# class2 = data[100:200, :]

# Synthetic data generation to match means
np.random.seed(42)  # For reproducibility
mean1 = np.array([6.381, 2.914, 4.2381, 1.3019])
mean2 = np.array([6.6147, 2.8968, 5.6256, 2.0478])
cov = np.eye(4)  # Assume identity covariance for simplicity (adjust if needed)

class1_features = multivariate_normal.rvs(mean=mean1, cov=cov, size=100)
class1 = np.column_stack((class1_features, np.ones(100)))  # Add label 1 in column 4

class2_features = multivariate_normal.rvs(mean=mean2, cov=cov, size=100)
class2 = np.column_stack((class2_features, np.full(100, 2)))  # Add label 2 in column 4

data = np.vstack((class1, class2))

# ฟังก์ชันคำนวณ Mean Vector
def calculate_mean(data_subset):
    return np.mean(data_subset, axis=0)

# ฟังก์ชันคำนวณ Covariance Matrix (ML)
def calculate_covariance(data_subset, mean):
    n_samples = data_subset.shape[0]
    centered_data = data_subset - mean
    return np.dot(centered_data.T, centered_data) / n_samples

# ฟังก์ชันจำแนกด้วย Bayes
def classify(x, mean1, mean2, cov1, cov2):
    dist1 = multivariate_normal(mean=mean1, cov=cov1, allow_singular=True)
    dist2 = multivariate_normal(mean=mean2, cov=cov2, allow_singular=True)
    p1 = dist1.pdf(x)
    p2 = dist2.pdf(x)
    if p1 > p2:
        return 1
    elif p1 < p2:
        return 2
    else:
        return np.random.choice([1, 2])

# ฟังก์ชันทดสอบหลัก (output เป็น HTML)
def test_model(class1, class2, features, feature_desc):
    n_features = len(features)
    feat_names = [f + 1 for f in features]
    html_output = f"""
    <html><head><title>Bayes Classifier Results</title>
    <style> table {{ border-collapse: collapse; }} th, td {{ border: 1px solid black; padding: 5px; }} </style></head><body>
    <h2>การทดสอบกับ {feature_desc} (Features: {', '.join(map(str, feat_names))})</h2>
    """

    X1 = class1[:, features]
    X2 = class2[:, features]
    Y1 = class1[:, 4]
    Y2 = class2[:, 4]

    accuracies = []
    for test_id in range(1, 12):
        if test_id < 11:
            start_idx = (test_id - 1) * 10
            end_idx = test_id * 10
            train1 = np.vstack((class1[:start_idx], class1[end_idx:]))
            train2 = np.vstack((class2[:start_idx], class2[end_idx:]))
            test1 = class1[start_idx:end_idx]
            test2 = class2[start_idx:end_idx]
            test_desc = f"Test {test_id}: Rows {start_idx+1}-{end_idx}"
        else:
            train1 = class1
            train2 = class2
            test1 = class1
            test2 = class2
            test_desc = f"Test {test_id}: All data (1-100)"

        mean1 = calculate_mean(train1[:, features])
        mean2 = calculate_mean(train2[:, features])
        cov1 = calculate_covariance(train1[:, features], mean1)
        cov2 = calculate_covariance(train2[:, features], mean2)

        # Mean table as HTML
        mean_table = [
            ["Mean w1"] + [f"{x:.4f}" for x in mean1],
            ["Mean w2"] + [f"{x:.4f}" for x in mean2]
        ]
        html_output += f"<h3>{test_desc}</h3><h4>Mean Vectors:</h4>{tabulate(mean_table, headers=[''] + [f'Feature {i+1}' for i in range(n_features)], tablefmt='html')}"

        # Cov table as HTML (row-major)
        cov1_flat = [f"{cov1[i,j]:.4f}" for i in range(n_features) for j in range(n_features)]
        cov2_flat = [f"{cov2[i,j]:.4f}" for i in range(n_features) for j in range(n_features)]
        cov_headers = [f"({i+1},{j+1})" for i in range(n_features) for j in range(n_features)]
        cov1_table = [["Cov w1"] + cov1_flat]
        cov2_table = [["Cov w2"] + cov2_flat]
        html_output += f"<h4>Covariance Matrices (Row-wise):</h4>{tabulate(cov1_table, headers=[''] + cov_headers, tablefmt='html')}{tabulate(cov2_table, headers=[''] + cov_headers, tablefmt='html')}"

        # Classify
        pred1 = [classify(test1[i, features], mean1, mean2, cov1, cov2) for i in range(len(test1))]
        pred2 = [classify(test2[i, features], mean1, mean2, cov1, cov2) for i in range(len(test2))]
        actual_pred = np.array(pred1 + pred2)
        truth = np.concatenate([test1[:, 4], test2[:, 4]])

        # Confusion Matrix
        conf = np.zeros((2, 2), dtype=int)
        for p, t in zip(actual_pred, truth):
            conf[int(t) - 1, int(p) - 1] += 1

        actual_table = [
            ["Actual \\ Pred"] + ["Class 1", "Class 2"],
            ["Class 1"] + [str(conf[0,0]), str(conf[0,1])],
            ["Class 2"] + [str(conf[1,0]), str(conf[1,1])]
        ]
        html_output += f"<h4>Actual Table:</h4>{tabulate(actual_table, tablefmt='html')}"

        acc = np.sum(actual_pred == truth) / len(truth) * 100
        html_output += f"<p><strong>Accuracy (Correct) = {acc:.2f}%</strong></p><br><br><br>"
        accuracies.append(acc)

    avg_acc = np.mean(accuracies)
    html_output += f"<h3>> Average Accuracy: {avg_acc:.2f}%</h3></body></html>"
    return html_output

# Main: ถาม features จาก input
if __name__ == "__main__":
    # Simulate input for demo; in real run, use input()
    # features_input = input("Enter features (1-based, comma separated, e.g., 1,2,3,4 for all): ")
    # features = [int(f.strip()) - 1 for f in features_input.split(',')]

    # Demo with 4 features
    features = [0,1,2,3]
    print(test_model(class1, class2, features, "4 Features (Demo)"))

    # Demo with 2 features (1 and 2)
    features = [0,1]
    print(test_model(class1, class2, features, "2 Features (1 and 2)"))

    # ถ้าต้องการ run ด้วย input จริง, uncomment และ run in terminal