import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import pathlib
import os
import numpy as np
from sklearn.utils import class_weight

# --- 1. PREPARE THE DATASET ---
data_dir = pathlib.Path("E:/VisionGuide/Data/Dataset/dataset")

# --- 2. PREPROCESS THE DATA ---
IMG_SIZE = 224
BATCH_SIZE = 32

# Create datasets for training and validation
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE)

class_names = train_ds.class_names
print(f"Detected class names: {class_names}")

# --- CALCULATE CLASS WEIGHTS ---
# Get the labels from the training dataset
labels = np.concatenate([y for x, y in train_ds], axis=0)
# Compute weights to handle imbalance
class_weights_dict = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(labels),
    y=labels
)
# Convert to a dictionary with integer keys for model.fit()
class_weights = {i: weight for i, weight in enumerate(class_weights_dict)}
print(f"Calculated class weights: {class_weights}")

# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. BUILD THE DEEP LEARNING MODEL ---
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
data_augmentation = Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False

model = Sequential([
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(len(class_names))
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# --- 4. TRAIN THE MODEL (PHASE 1) ---
epochs_phase_1 = 20
print("\n--- PHASE 1: Training top layers with class weights ---")
history_phase_1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs_phase_1,
    class_weight=class_weights  # Apply class weights here
)

# --- 5. FINE-TUNE THE ENTIRE MODEL (PHASE 2) ---
base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              metrics=['accuracy'])

model.summary()

epochs_phase_2 = 20
total_epochs = epochs_phase_1 + epochs_phase_2

early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

print("\n--- PHASE 2: Fine-tuning entire model with class weights ---")
history_phase_2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,
    initial_epoch=history_phase_1.epoch[-1],
    callbacks=[early_stopping],
    class_weight=class_weights  # Apply class weights here as well
)

# --- 6. SAVE THE TRAINED MODEL ---
model_folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Model')
if not os.path.exists(model_folder_path):
    os.makedirs(model_folder_path)

model_save_path = os.path.join(model_folder_path, 'model.h5')
model.save(model_save_path)

print(f"\nTraining finished successfully!")
print(f"New model saved to: {model_save_path}")
print("Please copy this new model to your Flask application's model directory and test it.")