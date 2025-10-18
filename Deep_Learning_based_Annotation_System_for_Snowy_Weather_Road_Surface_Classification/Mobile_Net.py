import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Dataset parameters
data_dir = '/kaggle/input/snow-dataset/classification'  # Main directory containing class folders
classes = ['clear', 'light', 'medium', 'plowed']
img_size = (224, 224)  # MobileNet default input size
batch_size = 32

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Using 20% for validation
)

# Calculate class weights
def get_class_weights(directory):
    # Get list of all image file paths
    file_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(directory, class_name)
        class_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        file_paths.extend(class_files)
        labels.extend([class_idx] * len(class_files))
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return dict(enumerate(class_weights))

class_weights = get_class_weights(data_dir)
print("Class weights:", class_weights)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Load MobileNetV2 base model (pre-trained on ImageNet)
base_model = MobileNetV2(
    input_shape=(img_size[0], img_size[1], 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model
base_model.trainable = False

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')  # 4 output classes
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=3)
]

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    epochs=10,
    callbacks=callbacks,
    class_weight=class_weights
)

# Evaluate the model
print("\nTraining Accuracy:")
train_loss, train_acc = model.evaluate(train_generator)
print(f"Training accuracy: {train_acc:.4f}")

print("\nValidation Accuracy:")
val_loss, val_acc = model.evaluate(val_generator)
print(f"Validation accuracy: {val_acc:.4f}")

# Print class-wise accuracy
print("\nClass-wise Validation Accuracy:")
val_pred = model.predict(val_generator)
val_pred_classes = np.argmax(val_pred, axis=1)
val_true_classes = val_generator.classes

from sklearn.metrics import classification_report
print(classification_report(val_true_classes, val_pred_classes, target_names=classes))