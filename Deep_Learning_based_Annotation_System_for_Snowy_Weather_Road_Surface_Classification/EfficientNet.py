import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = "/kaggle/input/snow-dataset/classification"  # Folder containing subfolders: clear, light, medium, plowed

# Load the dataset
train_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Class names and mapping
class_names = train_ds.class_names
num_classes = len(class_names)

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Calculate class weights
def get_class_weights(ds):
    labels = []
    for _, y in ds.unbatch():
        labels.append(y.numpy())
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    return dict(enumerate(class_weights))

class_weights = get_class_weights(train_ds)

# Create EfficientNet model
base_model = EfficientNetB3(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Freeze base model
base_model.trainable = False

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weights
)

# Unfreeze and fine-tune
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=5, class_weight=class_weights)

# Evaluation
val_labels = []
val_preds = []

for images, labels in val_ds:
    preds = model.predict(images)
    val_preds.extend(np.argmax(preds, axis=1))
    val_labels.extend(labels.numpy())

# Print metrics
print("Classification Report:")
print(classification_report(val_labels, val_preds, target_names=class_names))
