import os
import numpy as np
import rasterio
import xml.etree.ElementTree as ET
import tensorflow as tf
from keras import layers, models, regularizers
from keras.models import load_model
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split , KFold
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
import time

COMMON_WIDTH, COMMON_HEIGHT = 128, 128  # Example dimensions

# Data Loading
def load_sar_image(file_path):
    with rasterio.open(file_path, 'r') as src:
        sar_image = src.read()
    return sar_image

# Preprocessing
def preprocess_sar_image(sar_image):
    vh_real = sar_image[0]
    vh_imag = sar_image[1]
    vv_real = sar_image[2]
    vv_imag = sar_image[3]
    
    vh_complex = vh_real + 1j * vh_imag
    vv_complex = vv_real + 1j * vv_imag
    
    vh_amplitude = np.abs(vh_complex)
    vv_amplitude = np.abs(vv_complex)
    
    vh_phase = np.angle(vh_complex)
    vv_phase = np.angle(vv_complex)
    
    amplitude = (vh_amplitude + vv_amplitude) / 2
    phase = (vh_phase + vv_phase) / 2

     # Resizing
    amplitude = resize(amplitude, (COMMON_WIDTH, COMMON_HEIGHT))
    phase = resize(phase, (COMMON_WIDTH, COMMON_HEIGHT))

    preprocessed_image = np.stack([amplitude, phase], axis=-1)
   #print("Preprocessed image shape: ", preprocessed_image.shape)
     # Resizing
    '''
    # 4 Channel input
    vh_real = resize(vh_real, (COMMON_WIDTH, COMMON_HEIGHT))
    vh_imag = resize(vh_imag, (COMMON_WIDTH, COMMON_HEIGHT))
    vv_real = resize(vv_real, (COMMON_WIDTH, COMMON_HEIGHT))
    vv_imag = resize(vv_imag, (COMMON_WIDTH, COMMON_HEIGHT))

    preprocessed_image = np.stack([vh_real, vh_imag, vv_real, vv_imag], axis=-1)
    '''
    return preprocessed_image

def process_all_images_in_folder(folder_path):
    preprocessed_images = []
    filenames = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".tif"):
            file_path = os.path.join(folder_path, file_name)
            raw_sar_image = load_sar_image(file_path)
            preprocessed_image = preprocess_sar_image(raw_sar_image)
            preprocessed_images.append(preprocessed_image)
            filenames.append(file_name)
    return preprocessed_images, filenames

# Model Definition

def model_arch_19(input_shape, num_classes):
    
    model = models.Sequential()
 
    # Convolutional Layer
    model.add(layers.Conv2D(16, (5, 5), strides=(1,1), padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())  
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(layers.Conv2D(32, (5, 5), strides=(1,1), padding='same'))
    model.add(layers.BatchNormalization())  
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    model.add(layers.Conv2D(64, (5, 5),strides=(1,1),padding='same'))
    model.add(layers.BatchNormalization())  
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(layers.Conv2D(128, (5, 5),strides=(1,1),padding='same'))
    model.add(layers.BatchNormalization())  
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(layers.Conv2D(256, (5, 5),strides=(1,1),padding='same'))
    model.add(layers.BatchNormalization())  
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(layers.Conv2D(512, (5, 5),strides=(1,1),padding='same'))
    model.add(layers.BatchNormalization())  
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(layers.Conv2D(1028, (5, 5),strides=(1,1),padding='same'))
    model.add(layers.BatchNormalization())  
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(2048,kernel_regularizer=l2(0.1),activation='relu'))
    model.add(Dropout(0.25))
    model.add(layers.Dense(128,kernel_regularizer=l2(0.1),activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

def model_arch_20(input_shape, num_classes):
    
    model = models.Sequential()
 
     # Convolutional Layer
    model.add(layers.Conv2D(16, (5, 5), strides=(1,1), padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())  
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(layers.Conv2D(32, (5, 5), strides=(1,1), padding='same'))
    model.add(layers.BatchNormalization())  
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    model.add(layers.Conv2D(64, (5, 5),strides=(1,1),padding='same'))
    model.add(layers.BatchNormalization())  
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(layers.Conv2D(128, (5, 5),strides=(1,1),padding='same'))
    model.add(layers.BatchNormalization())  
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(layers.Conv2D(256, (5, 5),strides=(1,1),padding='same'))
    model.add(layers.BatchNormalization())  
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(layers.Conv2D(512, (5, 5),strides=(1,1),padding='same'))
    model.add(layers.BatchNormalization())  
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(2560,kernel_regularizer=l2(0.1),activation='relu'))
    model.add(Dropout(0.25))
    model.add(layers.Dense(128,kernel_regularizer=l2(0.1),activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

def model_arch_18(input_shape, num_classes):

    model = models.Sequential()
 
    # Convolutional Layer
    model.add(layers.Conv2D(16, (5, 5), strides=(1,1), padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())  
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(layers.Conv2D(32, (5, 5), strides=(1,1), padding='same'))
    model.add(layers.BatchNormalization())  
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    model.add(layers.Conv2D(64, (5, 5),strides=(1,1),padding='same'))
    model.add(layers.BatchNormalization())  
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(layers.Conv2D(128, (5, 5),strides=(1,1),padding='same'))
    model.add(layers.BatchNormalization())  
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(layers.Conv2D(256, (5, 5),strides=(1,1),padding='same'))
    model.add(layers.BatchNormalization())  
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(layers.Conv2D(512, (5, 5),strides=(1,1),padding='same'))
    model.add(layers.BatchNormalization())  
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(layers.Conv2D(1028, (5, 5),strides=(1,1),padding='same'))
    model.add(layers.BatchNormalization())  
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(256,kernel_regularizer=l2(0.1),activation='relu'))
    model.add(Dropout(0.25))
    model.add(layers.Dense(128,kernel_regularizer=l2(0.1),activation='relu'))
    model.add(Dropout(0.25))
    model.add(layers.Dense(num_classes, activation='softmax'))
   
    return model


model_archs = [model_arch_19,model_arch_20] # Add more models here

all_images = []
all_labels = []

# Main Script
main_dataset_path = "Modified_OpenSARShip\Modified_OpenSARShip"
for folder in os.listdir(main_dataset_path):
    data_folder = os.path.join(main_dataset_path, folder)
    if folder.startswith('.') or not os.path.isdir(data_folder):
        continue
    xml_path = os.path.join(data_folder, "Ship.xml")
    patch_folder = os.path.join(data_folder, "Patch")
    images, filenames = process_all_images_in_folder(patch_folder)
    if not images:
        print("No images were processed!")
        exit()
    all_images.extend(images)
    all_labels.extend([filename.split('_')[0] for filename in filenames])  
    
X = np.array(all_images)
# Check dimensions
if len(X.shape) < 4:
    print("Error in image dimensions.")
    exit()

    
unique_labels = np.unique(all_labels)

label_to_int = {label: i for i, label in enumerate(unique_labels)}
y_int = np.array([label_to_int[label] for label in all_labels])
np.save('best_unique_labels.npy', unique_labels)
unique_labels, counts = np.unique(y_int, return_counts=True)
label_count_dict = dict(zip(unique_labels, counts))

for label, count in label_count_dict.items():
    print(f"Class Label: {label}, Count: {count}")

X = X.astype('float32') / 255.0
y_encoded = to_categorical(y_int)

# Split the data into training+validation and test sets
#X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_encoded, test_size=0.1, random_state=42)
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

save_dir = "Saved_models"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model_paths = []

# Training loop for each architecture

model_training_metrics = {}

with tf.device("/gpu:0"):
    for model_func in model_archs:
        model_name = model_func.__name__
        print(f"\nTraining model: {model_name}")
        
        training_losses = []
        training_accuracies = []
        training_times = []
        validation_losses = []
        validation_accuracies = []
        inference_times = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            print(f"Training on fold {fold}...")

            # Split the data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train_encoded, y_val_encoded = y_encoded[train_idx], y_encoded[val_idx]
            
            # Create and compile the model
            model = model_func((X.shape[1], X.shape[2], 2), len(unique_labels))
            optimizer = Adam(learning_rate=0.0001)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=0.00001)
            early_stop = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)
            
            # Train the model
            start_time = time.time()
            history = model.fit(X_train, y_train_encoded, batch_size=4, epochs=300, steps_per_epoch=len(X_train) // 4, validation_data=(X_val, y_val_encoded), callbacks=[early_stop, reduce_lr])
            end_time = time.time()
            training_time = end_time - start_time
            training_times.append(training_time)

            start_time2 = time.time()
            _ = model.predict(X_val)
            end_time2 = time.time()

            inference_time = end_time2 - start_time2
            inference_times.append(inference_time / len(X_val))

            # Store training loss and accuracy for this fold
            training_losses.append(history.history['loss'][-1])
            training_accuracies.append(history.history['accuracy'][-1])
            validation_losses.append(history.history['val_loss'][-1]) 
            fold_accuracy = max(history.history['val_accuracy'])
            validation_accuracies.append(fold_accuracy)

            # Save model
            model_path = os.path.join(save_dir, f"{model_name}_fold_{fold}.h5")
            model_paths.append(model_path)
            model.save(model_path)

        average_inference_time = np.mean(inference_times)
        average_validation_accuracy = np.mean(validation_accuracies)
        average_validation_loss = np.mean(validation_losses)
        average_training_loss = np.mean(training_losses)
        average_training_accuracy = np.mean(training_accuracies)
        average_training_time = np.mean(training_times)
        formatted_average_inference_time = f"{average_inference_time * 1000:.4f}"  # Convert to milliseconds
        model_training_metrics[model_name] = {
        "Average Inference Time per Image (Miliseconds)": formatted_average_inference_time,
        "Training Loss": average_training_loss,
        "Training Accuracy": average_training_accuracy,
        "Training Time": average_training_time,
        "Validation Loss": average_validation_loss, 
        "Validation Accuracy": average_validation_accuracy
        }

        

print("\nSummary of Model Performances:")
for model_name, metrics in model_training_metrics.items():
    print(f"\nModel: {model_name}")
    for metric_name, value in metrics.items():
        if metric_name == "Training Time":
            print(f"{metric_name}: {value:.2f} Seconds")
        elif metric_name == "Average Inference Time per Image (Miliseconds)":
            print(f"{metric_name}: {value} Milliseconds")
        else:
            print(f"{metric_name}: {value:.4f}")
