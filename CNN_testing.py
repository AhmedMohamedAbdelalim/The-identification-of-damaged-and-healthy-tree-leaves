import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

# Load the saved model
model = tf.keras.models.load_model("cnn_model.h5")

# Define the function to load and preprocess the test images
def load_and_preprocess_images(folder_path):
    image_list = []
    image_labels = []
    label_counter = -1

    # Iterate over the subfolders in the parent directory
    for folder_1 in os.listdir(folder_path):
        folder_1_path = os.path.join(folder_path, folder_1)
        if os.path.isdir(folder_1_path):
            # Iterate over the nested subfolders
            for folder_2 in os.listdir(folder_1_path):
                folder_2_path = os.path.join(folder_1_path, folder_2)
                if os.path.isdir(folder_2_path):
                    label_counter += 1
                    # Iterate over the files in the nested folder
                    for filename in os.listdir(folder_2_path):
                        if filename.endswith(".JPG") or filename.endswith(".png"):
                            file_path = os.path.join(folder_2_path, filename)
                            image = Image.open(file_path).convert("RGB")
                            image = image.resize((224, 224))  # Resize the image to match the input size of the model
                            image = np.array(image) / 255.0  # Normalize pixel values to range [0, 1]
                            image_list.append(image)
                            image_labels.append(label_counter)

    return np.array(image_list), np.array(image_labels)

# Define the path to the test images
test_folder_path = 'E:/semester6/Machine Learning/project/Test_imge'

# Load and preprocess the test images
test_images, test_labels = load_and_preprocess_images(test_folder_path)

# Make predictions on the test images using the saved model
y_pred = model.predict(test_images)
y_pred_labels = np.argmax(y_pred, axis=1)

# Calculate and print accuracy, precision, recall, and F1 score for the test data
test_accuracy = accuracy_score(test_labels, y_pred_labels)
test_precision = precision_score(test_labels, y_pred_labels, average='weighted')
test_recall = recall_score(test_labels, y_pred_labels, average='weighted')
test_f1_score = f1_score(test_labels, y_pred_labels, average='weighted')

print("Testing Accuracy:", test_accuracy)
print("Precision:", test_precision)
print("Recall:", test_recall)
print("F1 Score:", test_f1_score)
