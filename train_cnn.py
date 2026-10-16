import tensorflow as tf 
from tensorflow.keras import layers, models
IMG = (96,96)
train = tf.keras.utils.image_dataset_from_directory("data_split/train", image_size = IMG, batch_size =16, label_mode = "binary")
val = tf.keras.utils.image_dataset_from_directory("data_split/val", image_size = IMG, batch_size = 16, label_mode = "binary")

model = models.Sequential([
    layers.Input(shape = (96,96,3)),
    layers.Rescaling(1./225),
    layers.Conv2D(32, 3, activation = "relu"), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation = "relu"), layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation = "relu"), layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(64, activation = "relu"),
    layers.Dense(1, activation = "sigmoid"),
    
])
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.fit(train, validation_data = val, epochs = 10)
model.save("cnn_simple.keras")