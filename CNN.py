import glob
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import os
import tensorflow as tf
import joblib

# Set the GPU memory limit
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Define the directory path where your images are located
parent_folder_path = 'E:/semester6/Machine Learning/project/archive (2)'

image_list = []
image_labels = []
x = -1
# Iterate over the subfolders in the parent directory
for folder_1 in os.listdir(parent_folder_path):
    folder_1_path = os.path.join(parent_folder_path, folder_1)
    if os.path.isdir(folder_1_path):
        # Iterate over the nested subfolders
        for folder_2 in os.listdir(folder_1_path):
            folder_2_path = os.path.join(folder_1_path, folder_2)
            if os.path.isdir(folder_2_path):
                x += 1
                # Iterate over the files in the nested folder
                for filename in os.listdir(folder_2_path):
                    if filename.endswith(".JPG") or filename.endswith(".png"):
                        file_path = os.path.join(folder_2_path, filename)
                        image = Image.open(file_path).convert("RGB")
                        image = image.resize((224, 224))  # Resize the image to match input size
                        image_list.append(np.array(image))
                        image_labels.append(x)

print(x)
# Convert the image data to a numpy array
X = np.array(image_list)

# Preprocess and encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(image_labels)
num_classes = len(label_encoder.classes_)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split the training set into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Normalize pixel values to range [0, 1]
x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0

# Create TensorFlow Dataset objects
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# Define the CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Add early stopping callback
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
model.fit(train_dataset, epochs=50, validation_data=val_dataset, callbacks=[early_stopping])

# Save the trained model
model.save("cnn_model.h5")

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Make predictions on the test dataset
y_pred = model.predict(test_dataset)
y_pred_labels = np.argmax(y_pred, axis=1)

# Decode the predicted labels back to their original text form
y_pred_labels_text = label_encoder.inverse_transform(y_pred_labels)

# Print the predicted labels
for i in range(len(x_test)):
    print("Image {}: Actual: {}, Predicted: {}".format(i+1, label_encoder.classes_[y_test[i]], y_pred_labels_text[i]))

print("______________________________________________________________")
# Calculate and print accuracy, f-score, recall, and precision
accuracy = accuracy_score(y_test, y_pred_labels)
print("Accuracy: ", accuracy)
