import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import random

# --- Configuration ---
DATA_DIR = 'Dataset' 
MODEL_PATH = 'Model/model.h5' 
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10  # Reduced for demonstration; increase for better accuracy
# Split 80% for training, 20% for validation
TRAIN_SPLIT = 0.8 

# --- New Function: Automatic Data Splitting ---
def setup_dataset_structure(data_dir, train_split):
    """
    Checks if train/val directories exist and performs a split if only
    class directories are found directly under the DATA_DIR.
    """
    
    # 1. Check if the structure is already correct (i.e., 'train' exists)
    if os.path.isdir(os.path.join(data_dir, 'train')) and \
       os.path.isdir(os.path.join(data_dir, 'val')):
        print("Dataset structure already set up (found 'train' and 'val' folders). Skipping split.")
        return

    print("Current structure is flat. Performing automatic Train/Val split...")
    
    # Define new directories
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Find all class folders (your cataract, normal, etc.)
    class_names = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(('.', '__'))
                   and d not in ['train', 'val']]

    if not class_names:
        raise FileNotFoundError(f"No class folders found directly in '{data_dir}'.")

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        images = os.listdir(class_path)
        random.shuffle(images)
        
        # Calculate split index
        split_index = int(len(images) * train_split)
        train_images = images[:split_index]
        val_images = images[split_index:]

        # Create destination class folders
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        # Move files
        for img in train_images:
            shutil.move(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
        
        for img in val_images:
            shutil.move(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))
            
        # Clean up the original class folder if it's empty
        if not os.listdir(class_path):
             shutil.rmtree(class_path)
        
        print(f"Split {class_name}: {len(train_images)} train, {len(val_images)} val.")

# --- Data Preparation ---

print("Step 1: Preparing Data Generators...")

try:
    # RUN THE NEW SETUP FUNCTION BEFORE INITIALIZING GENERATORS
    setup_dataset_structure(DATA_DIR, TRAIN_SPLIT)
except Exception as e:
    print("\n--- FATAL ERROR DURING DATA SETUP ---")
    print(f"Details: {e}")
    exit()

# Data Augmentation and Rescaling for Training
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values to 0-1
    rotation_range=20,       # Rotate images up to 20 degrees
    width_shift_range=0.2,   
    height_shift_range=0.2,  
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Rescaling only for Validation (no augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)

# Load data from the newly structured directories
try:
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'val'),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

except Exception as e:
    print("\n--- ERROR: Data Directory Issue (After Split) ---")
    print("Something went wrong loading data even after the automatic split. Check contents.")
    print(f"Details: {e}")
    exit()

# Determine number of classes (needed for the output layer)
NUM_CLASSES = train_generator.num_classes
CLASS_NAMES = list(train_generator.class_indices.keys())
print(f"Identified {NUM_CLASSES} classes: {CLASS_NAMES}")

# --- Model Definition (A Simple CNN Architecture) ---

print("\nStep 2: Defining the CNN Model...")

model = Sequential([
    # First Conv Block
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPooling2D(2, 2),

    # Second Conv Block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Third Conv Block
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Flatten and Dense Layers
    Flatten(),
    Dropout(0.5), # Regularization to prevent overfitting
    Dense(512, activation='relu'),

    # Output Layer: Softmax for multi-class classification
    Dense(NUM_CLASSES, activation='softmax')
])

# --- Model Compilation ---

print("\nStep 3: Compiling the Model...")

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- Model Training ---

print("\nStep 4: Starting Model Training...")

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# --- Model Saving ---

print("\nStep 5: Saving the Trained Model...")

try:
    # Ensure the Model directory exists before saving
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Save the model
    model.save(MODEL_PATH)
    print(f"Successfully trained and saved model to {MODEL_PATH}")

    # Optionally save class names so Flask app knows the order
    class_names_path = 'class_names.txt'
    with open(class_names_path, 'w') as f:
        f.write('\n'.join(CLASS_NAMES))
    print(f"Saved class names to {class_names_path}")

except Exception as e:
    print(f"Error saving model: {e}")

print("\nTraining complete.")
# The model.h5 file is now ready for use by Web/app.py
