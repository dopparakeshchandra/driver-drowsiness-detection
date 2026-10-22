import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

test = tf.keras.utils.image_dataset_from_directory(
    "data_split/test", image_size=(96,96), batch_size = 32,
    label_mode = "binary", shuffle = False)
model = tf.keras.models.load_model("mobilenet.keras")

y_true = np.concatenate([y.numpy() for _, y in test]).ravel()
y_pred = (model.predict(test) > 0.5).astype(int).ravel()

report = classification_report (y_true, y_pred, target_names = test.class_names, digits = 4)
cm = confusion_matrix(y_true, y_pred)
print(report)
print(cm)

with open("result_mobilenet.txt", "w") as  f:
    f.write(report + "\n" + str(cm) + "\n")