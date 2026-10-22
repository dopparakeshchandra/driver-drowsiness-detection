import tensorflow as tf 
from tensorflow.keras import layers, models 

IMG = (96,96)
BATCH = 32
AUTOTUNE = tf.data.AUTOTUNE

train = tf.keras.utils.image_dataset_from_directory(
    "data_split/train", image_size = IMG, batch_size = BATCH, label_mode = "binary")
val = tf.keras.utils.image_dataset_from_directory(
    "data_split/val", image_size = IMG, batch_size = BATCH, label_mode = "binary")
train = train.prefetch(AUTOTUNE)
val = val.prefetch(AUTOTUNE)

base = tf.keras.applications.MobileNetV2(input_shape = (96,96,3), include_top = False, weights = "imagenet")
base.trainable = False

input = layers.Input(shape=(96,96,3))
x = layers.Rescaling(1./127.5, offset = -1)(input)
x = base(x, training = False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(1, activation = "sigmoid")(x)
model = models.Model(input, output)

model.compile(optimizer = tf.keras.optimizers.Adam(1e-3),
              loss = "binary_crossentropy", metrics = ["accuracy"])
print("=== Phase 1: frozen base ===")
model.fit(train, validation_data = val, epochs = 3)


base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
model.compile(optimizer = tf.keras.optimizers.Adam(1e-5),
              loss="binary_crossentropy", metrics = ["accuracy"])
print("=== Phase 2: fine-tuningg ===")
model.fit(train, validation_data = val, epochs = 2)

model.save("mobilenet.keras")