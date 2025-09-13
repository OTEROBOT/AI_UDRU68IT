import numpy as np
import pandas as pd

# โหลดข้อมูล
df = pd.read_csv("heart.csv")

# สมมติว่าคอลัมน์ target คือ class
class0 = df[df['target'] == 0]
class1 = df[df['target'] == 1]

# คำนวณ mean และ variance ของแต่ละ feature
mean0 = class0.mean(numeric_only=True)
mean1 = class1.mean(numeric_only=True)

var0 = class0.var(numeric_only=True)
var1 = class1.var(numeric_only=True)

# เก็บผลลัพธ์
results = {}

for feature in df.drop(columns=['target']).columns:
    mu_i, mu_j = mean0[feature], mean1[feature]
    var_i, var_j = var0[feature], var1[feature]

    # คำนวณ d_ij ตามสูตร
    d = 0.5 * ((var_j / var_i) + (var_i / var_j) - 2) \
        + 0.5 * ((mu_i - mu_j) ** 2) * ((1 / var_i) + (1 / var_j))

    results[feature] = d

# แปลงเป็น DataFrame และเรียงจากมาก -> น้อย
df_result = pd.DataFrame(list(results.items()), columns=['Feature', 'Distance'])
df_result = df_result.sort_values(by="Distance", ascending=False)

print(df_result)