import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

test = tf.keras.utils.image_dataset_from_directory("data_split/test", image_size =(96,96), batch_size = 16, label_mode = "binary", shuffle = False)
model = tf.keras.models.load_model("cnn_simple.keras")

y_true = np.concatenate([y.numpy() for _, y in test]).ravel()
y_pred = np.concatenate(model.predict(test) > 0.5).astype(int).ravel()

report = classification_report(y_true, y_pred, target_names = ['awake', 'sleepy'])
cm = confusion_matrix(y_true, y_pred)
print(report)
print(cm)

with open("result.txt","w") as f:
    f.write("Test set results (12,735 images)\n\n")
    f.write(report)
    f.write("\nConfusion matrix (rows = true, columns = predicted)\n")
    f.write(f"{cm}\n")