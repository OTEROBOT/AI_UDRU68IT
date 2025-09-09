import numpy as np
from sklearn.preprocessing import StandardScaler

# โหลด dataset
data = np.genfromtxt('haberman.csv', delimiter=',', skip_header=1)

train_data = data[:, :3]
train_labels = data[:, 3].astype(int)

# --- Normalize (StandardScaler) ---
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)

# Input test data
age = float(input("Enter patient's age at time of operation (numerical): "))
year = float(input("Enter patient's year of operation (19xx, numerical): "))
nodes = float(input("Enter number of positive axillary nodes detected (numerical): "))

test_data_original = np.array([age, year, nodes])   # เก็บค่าจริง
test_data = scaler.transform([test_data_original])[0]   # normalize

# Set K
K = 5

# Calculate Euclidean distances
distances = np.sqrt(np.sum((train_data - test_data)**2, axis=1))
indices = np.argsort(distances)
nearest_classes = train_labels[indices[:K]]

# Count class
class_counts = np.bincount(nearest_classes)
predicted_class = np.argmax(class_counts)

# Output
print("\nInput Test Data:", (float(age), float(year), float(nodes)))
print("Predicted Class:", predicted_class)
if predicted_class == 1:
    print("The patient survived 5 years or longer.")
else:
    print("The patient died within 5 years.")
#python LAB4.py