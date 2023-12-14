import os
import numpy as np
import rasterio
import xml.etree.ElementTree as ET
import tensorflow as tf
from keras import layers, models, regularizers
from keras.models import load_model
from keras.layers import Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD , RMSprop  , Adadelta , Adagrad , Adamax , Nadam , Ftrl
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split , KFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from keras.preprocessing.image import ImageDataGenerator
import time

COMMON_WIDTH, COMMON_HEIGHT = 128, 128  # Example dimensions

def plot_learning_curve(history, metric, fold_number, save_dir):
   
    train_values = history.history[metric]
    val_values = history.history['val_' + metric]

    epochs = range(1, len(train_values) + 1)

    plt.figure()
    plt.plot(epochs, train_values, label='Training ' + metric)
    plt.plot(epochs, val_values, label='Validation ' + metric)
    plt.title(f'Training and Validation {metric} for fold {fold_number}')
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()

    file_name = f"{metric}_curve_fold_{fold_number}.png"
    plt.savefig(os.path.join(save_dir, file_name))
    plt.close()

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

def create_ap_cnn(input_shape, num_classes):
    
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
    model.add(layers.Dense(2048,kernel_regularizer=l2(0.1),activation='relu'))
    model.add(Dropout(0.25))
    model.add(layers.Dense(128,kernel_regularizer=l2(0.1),activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model
'''
datagen = ImageDataGenerator(
    rotation_range=15,           # Mild rotation
    width_shift_range=0.1,      # 10% shift
    height_shift_range=0.1,     # 10% shift
    zoom_range=0.05,            # 5% zoom in/out
    horizontal_flip=True,       # Horizontal flipping
    vertical_flip=True,         # Vertical flipping
    fill_mode='nearest',
)
'''
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
# X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_encoded, test_size=0.1, random_state=42)
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)


histories = []
saved_models = []
training_losses = []
training_accuracies = []
training_times = []
validation_losses = []
validation_accuracies = []
inference_times = []

fpr = dict()
tpr = dict()
roc_auc = dict()

save_dir = "Saved_model"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

plot_dir = "Learning_Curves"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

roc_curve_dir = "ROC_Curves"
if not os.path.exists(roc_curve_dir):
    os.makedirs(roc_curve_dir)

fold_number = 1
for train_index, val_index in kf.split(X):
    print(f"Training on fold {fold_number}...")
    
    # Split the data
    X_train, X_val = X[train_index], X[val_index]
    y_train_encoded, y_val_encoded = y_encoded[train_index], y_encoded[val_index]
    
    # Create and compile the model
    model = create_ap_cnn((X.shape[1], X.shape[2], 2), len(unique_labels))
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
    y_pred = model.predict(X_val)
    end_time2 = time.time()
    inference_time = end_time2 - start_time2
    inference_times.append(inference_time / len(X_val))

    for i in range(len(unique_labels)):
        fpr[i], tpr[i], _ = roc_curve(y_val_encoded[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i in range(len(unique_labels)):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Fold {fold_number}')
    plt.legend(loc='lower right')
    
    # Save the ROC curve plot to the roc_curve_dir
    roc_curve_file = os.path.join(roc_curve_dir, f"ROC_Curve_fold_{fold_number}.png")
    plt.savefig(roc_curve_file)
    plt.close()

     # Store training loss and accuracy
    training_losses.append(history.history['loss'][-1])
    training_accuracies.append(history.history['accuracy'][-1])
    validation_losses.append(history.history['val_loss'][-1]) 
    fold_accuracy = max(history.history['val_accuracy'])
    validation_accuracies.append(fold_accuracy)

    # Store the history and model
    histories.append(history)
    saved_models.append(model)
    
    # Save model
    model_path = os.path.join(save_dir, f"Best_Model_fold_{fold_number}.h5")
    model.save(model_path)
    
    fold_number += 1

print("K-fold cross-validation completed!")

average_inference_time = np.mean(inference_times)
formatted_average_inference_time = f"{average_inference_time * 1000:.4f}"  # Convert to milliseconds
average_validation_accuracy = np.mean(validation_accuracies)
average_validation_loss = np.mean(validation_losses)
average_training_loss = np.mean(training_losses)
average_training_accuracy = np.mean(training_accuracies)
average_training_time = np.mean(training_times)

print(f"Average Inference Time per Image (Miliseconds): {formatted_average_inference_time}")
print(f"Average Training Loss across all folds: {average_training_loss:.4f}")
print(f"Average Training Accuracy across all folds: {average_training_accuracy:.4f}")
print(f"Average Training Time across all folds: {average_training_time:.2f} seconds")
print(f"Average Validation Loss across all folds: {average_validation_loss:.4f}")
print(f"Average Validation Accuracy across all folds: {average_validation_accuracy:.4f}")

for fold_number, history in enumerate(histories, 1):
    plot_learning_curve(history, 'loss', fold_number, plot_dir)
    plot_learning_curve(history, 'accuracy', fold_number, plot_dir)



