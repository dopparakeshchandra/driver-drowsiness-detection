import os, shutil
from sklearn.model_selection import train_test_split
SRC = "data"
paths, labels = [], []
for cls in ["awake", "sleepy"]: 
    for f in os.listdir(os.path.join(SRC, cls)):
        paths.append(os.path.join(SRC, cls, f))
        labels.append(cls)
print("Total:", len(paths))

x_train, x_test, y_train, y_test = train_test_split(paths, labels, test_size = 0.30, stratify = labels,random_state = 42 )
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size = 0.50, stratify = y_test, random_state = 42)

for name, xs, ys in [("train", x_train, y_train), 
                     ("val", x_val, y_val),
                     ("test", x_test, y_test)]:
    for p, y in zip(xs, ys):
        dst = os.path.join("data_split", name, y)
        os.makedirs(dst, exist_ok =True)
        shutil.copy(p, dst)
    print(name, len(xs))    
    

for s in ["train", "val", "test"]:
    for c in ["awake", "sleepy"]:
        print(s, c, len(os.listdir(f"data_split/{s}/{c}")))