import numpy as np

a = np.array([1, 2, 3])
b = np.array([4,5,6])
c = np.stack((a,b), axis = 1)
print(c.shape)

TP = 200
FP = 50
TN = 650
FN = 100

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

import pandas as pd

data = [(1, "abc"), (2, "def"), (12, "ghi")]

df = pd.DataFrame(data, columns=["A", "B"])


print(df[(df['A'] > 5) & (df['B'].str.contains("abc"))])

new_df = df[df['A'] > 10]
print(new_df)