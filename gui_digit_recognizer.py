import tkinter as tk
from tkinter import Canvas, Button
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img # type: ignore
from PIL import Image


# Load the trained model
model = load_model('mnist.h5')

def predict_digit(image):
    # Preprocess image for model
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28 pixels
    image_array = img_to_array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    prediction = model.predict(image_array)
    return np.argmax(prediction[0])

def on_predict():
    image = canvas_to_image(canvas)
    digit = predict_digit(image)
    result_label.config(text=f'Predicted Digit: {digit}')

def canvas_to_image(canvas):
    canvas.postscript(file='temp.ps')
    img = Image.open('temp.ps')
    return img

# Create GUI
root = tk.Tk()
root.title("Digit Recognizer")

canvas = Canvas(root, width=280, height=280, bg='white')
canvas.pack()

predict_button = Button(root, text="Predict", command=on_predict)
predict_button.pack()

result_label = tk.Label(root, text="Predicted Digit: ")
result_label.pack()

root.mainloop()
