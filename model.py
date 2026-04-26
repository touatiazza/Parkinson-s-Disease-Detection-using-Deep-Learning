import tensorflow as tf
from keras.applications import DenseNet201
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
import time

# Image dimensions
IMG_HEIGHT, IMG_WIDTH = 299, 299
BATCH_SIZE = 16
NUM_CLASSES = 2
EPOCHS = 50
DATASET_DIR = r'path to data'

np.random.seed(42)
tf.random.set_seed(42)

def load_dataset(dataset_dir):
    image_paths = []
    labels = []
    class_names = ['HEALTHY', 'PD']
    
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"Class directory not found: {class_dir}")
            
        files = os.listdir(class_dir)
        if len(files) == 0:
            print(f"Warning: No files found in {class_dir}")
            continue
            
        for img_name in files:
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_name)
                image_paths.append(img_path)
                labels.append(class_idx)
    
    print(f"Found {len(image_paths)} images: {labels.count(0)} healthy, {labels.count(1)} Parkinson's")
    return np.array(image_paths), np.array(labels)

try:
    image_paths, labels = load_dataset(DATASET_DIR)
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    exit(1)

if len(image_paths) == 0:
    raise ValueError("No images found in the dataset.")

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths, labels, test_size=0.3, stratify=labels, random_state=42
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

train_df = pd.DataFrame({'filename': train_paths, 'class': train_labels.astype(str)})
val_df = pd.DataFrame({'filename': val_paths, 'class': val_labels.astype(str)})
test_df = pd.DataFrame({'filename': test_paths, 'class': test_labels.astype(str)})

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.densenet.preprocess_input
)
val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.densenet.preprocess_input
)
test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.densenet.preprocess_input
)

try:
    train_generator = train_datagen.flow_from_dataframe(
        train_df, x_col='filename', y_col='class', target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE, class_mode='binary', shuffle=True, workers=4, use_multiprocessing=False
    )
except Exception as e:
    print(f"Error creating training generator: {str(e)}")
    exit(1)

try:
    validation_generator = val_datagen.flow_from_dataframe(
        val_df, x_col='filename', y_col='class', target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE, class_mode='binary', shuffle=False, workers=4, use_multiprocessing=False
    )
except Exception as e:
    print(f"Error creating validation generator: {str(e)}")
    exit(1)

try:
    test_generator = test_datagen.flow_from_dataframe(
        test_df, x_col='filename', y_col='class', target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE, class_mode='binary', shuffle=False, workers=4, use_multiprocessing=False
    )
except Exception as e:
    print(f"Error creating test generator: {str(e)}")
    exit(1)

print(f"Class distribution in training set: {np.bincount(train_labels)}")
print(f"Class distribution in validation set: {np.bincount(val_labels)}")
print(f"Class distribution in test set: {np.bincount(test_labels)}")

class_counts = np.bincount(train_labels)
total_samples = np.sum(class_counts)
class_weights = {i: total_samples / (len(class_counts) * count) for i, count in enumerate(class_counts)}
print(f"Class weights: {class_weights}")

base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=outputs)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)
model_checkpoint = ModelCheckpoint('best_parkinson_densenet201_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
callbacks = [early_stopping, reduce_lr, model_checkpoint]

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

model.summary()

start_time = time.time()
print("Phase 1: Training with frozen feature extractor")
try:
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
except Exception as e:
    print(f"Error during training: {str(e)}")
    exit(1)

execution_time = time.time() - start_time
model.save('parkinson_densenet201_initial.h5')
print(f"Initial model saved to parkinson_densenet201_initial.h5, Execution time: {execution_time:.2f} seconds")

test_results = model.evaluate(test_generator, verbose=1)
print(f"Test loss: {test_results[0]:.4f}")
print(f"Test accuracy: {test_results[1]:.4f}")
print(f"Test AUC: {test_results[2]:.4f}")
print(f"Test Precision: {test_results[3]:.4f}")
print(f"Test Recall: {test_results[4]:.4f}")

def evaluate_model_in_detail(model, test_generator):
    try:
        test_generator.reset()
        y_true = test_generator.classes
        y_pred_prob = model.predict(test_generator, verbose=1)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        cm = confusion_matrix(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['HEALTHY', 'PARKINSON'], 
                   yticklabels=['HEALTHY', 'PARKINSON'])
        plt.title('Confusion Matrix (DenseNet201)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_densenet201.png')
        plt.close()
        print("Confusion matrix saved to confusion_matrix_densenet201.png")
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['HEALTHY', 'PARKINSON']))
        print(f"F1 Score: {f1:.4f}")
        
        return {'confusion_matrix': cm, 'f1_score': f1}
    except Exception as e:
        print(f"Error in evaluate_model_in_detail: {str(e)}")
        return None

try:
    eval_results = evaluate_model_in_detail(model, test_generator)
    if eval_results:
        print("Confusion matrix and F1 score generated successfully")
    else:
        print("Failed to generate confusion matrix and F1 score")
except Exception as e:
    print(f"Error evaluating model: {str(e)}")

def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(1, len(acc) + 1)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history_densenet201.png')
    plt.close()

try:
    plot_training_history(history)
    print("Training history plot saved to training_history_densenet201.png")
except Exception as e:
    print(f"Error plotting training history: {str(e)}")

model.save('parkinson_densenet201_model_final.h5')
print(f"Final model saved to parkinson_densenet201_model_final.h5, Execution time: {execution_time:.2f} seconds")

print("DenseNet201 model training and evaluation complete!")