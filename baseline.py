import os, numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def load(split):
    X,y = [], []
    for label, cls in enumerate(["awake", "sleepy"]):
        folder = f"data_split/{split}/{cls}"
        for f in os.listdir(folder):
            img = Image.open(os.path.join(folder, f)).convert("L").resize((64, 64))
            X.append(hog(np.array(img), orientations = 9,
                         pixels_per_cell = (8, 8), cells_per_block = (2, 2)))
            y.append(label)
    return np.array(X), np.array(y) 
X_train, y_train = load("train")
X_test, y_test = load("test")

clf = LogisticRegression(max_iter = 1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, target_names = ["awake", "sleepy"], digits = 4)
cm = confusion_matrix(y_test, y_pred)

print(report)
print(cm)

with open('Baseline_results.txt', "w") as f:
    f.write(report + "\n" + str(cm) +"\n")
    