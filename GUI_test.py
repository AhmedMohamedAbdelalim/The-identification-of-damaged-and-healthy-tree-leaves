import os
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import filedialog

# Load and preprocess a single test image
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))  # Resize the image to match input size
    image = np.array(image) / 255.0  # Normalize pixel values to range [0, 1]
    return image

# Load the saved model
model = keras.models.load_model("cnn_model.h5")

# Define the class labels
class_labels = ['Alstonia Scholaris disease', 'Alstonia Scholaris healthy','Arjun disease', 'Arjun healthy','Chinar disease', 'Chinar healthy','Gauva disease', 'Gauva healthy'
                ,'Jamun disease', 'Jamun healthy','Jatropha disease', 'Jatropha healthy'
                ,'Lemon disease', 'Lemon healthy','Mango disease', 'Mango healthy','Pomegranate disease', 'Pomegranate healthy','Pongamia Pinnata disease', 'Pongamia Pinnata healthy']

# Function to predict the class of a single image
def predict_image_class(image_path):
    image = load_and_preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction[0])
    class_label = class_labels[predicted_class]
    return class_label

# Function to print the image and its predicted class
def print_image_with_prediction(image_path):
    img = Image.open(image_path)

    # Predict the class label
    predicted_class = predict_image_class(image_path)

    # Create a blank image to draw the text
    text_image = Image.new('RGB', (img.width, img.height + 50), color=(255, 255, 255))
    text_image.paste(img, (0, 0))

    # Draw the predicted class label above the image
    draw = ImageDraw.Draw(text_image)
    font = ImageFont.truetype('arialbd.ttf', 50)  # Customize the font and size
    text_bbox = draw.textbbox((0, 0, img.width, 50), predicted_class, font=font)
    draw.rectangle([(0, 0), (img.width, 50)], fill=(0, 0, 255))
    draw.text(text_bbox[:2], predicted_class, fill=(255, 255, 255), font=font)

    # Show the image with the predicted class label
    text_image.show()

# Function to handle the button click event
def browse_image():
    # Open a file dialog to select an image file
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    # Print the image and its predicted class
    if image_path:
        print_image_with_prediction(image_path)

# Create the GUI window
window = tk.Tk()
window.title("Image Classification")
window.geometry("400x200")

# Create a button to browse for an image file
browse_button = tk.Button(window, text="Browse Image", command=browse_image)
browse_button.pack(pady=20)

# Run the GUI main loop
window.mainloop()
