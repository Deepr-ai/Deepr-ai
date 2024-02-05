import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from deeprai import models

network = models.FeedForward()
network.load("MNIST271k.deepr")
network.specs()

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.canvas_width = 200
        self.canvas_height = 200

        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack(side="left")

        self.canvas.bind("<B1-Motion>", self.paint)
        self.image1 = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image1)

        self.predict_button = tk.Button(root, text="Predict", command=self.predict)
        self.predict_button.pack(side="top")
        self.clear_button = tk.Button(root, text="Clear", command=self.clear)
        self.clear_button.pack(side="top")

        self.label = tk.Label(root, text="Draw a digit", font=("Helvetica", 16))
        self.label.pack(side="bottom")

    def paint(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)
        self.draw.ellipse([x1, y1, x2, y2], fill="black", outline="black")

    def clear(self):
        self.canvas.delete("all")
        self.image1 = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image1)
        self.label.config(text="Draw a digit")

    def predict(self):
        img = self.image1.resize((28, 28)).convert("L")
        img = ImageOps.invert(img)

        img_array = np.array(img).astype(np.float64) / 255.0
        img_array = img_array.reshape(-1)

        prediction = network.run(img_array)
        top_predictions = np.argsort(prediction)[-3:][::-1]

        self.label.config(
            text=f"Prediction: {top_predictions[0]} ({prediction[top_predictions[0]] * 100:.2f})")

root = tk.Tk()
app = DrawingApp(root)
root.title("Paint MNIST")
root.mainloop()
