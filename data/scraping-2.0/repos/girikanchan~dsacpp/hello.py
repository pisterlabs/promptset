import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import requests
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
import os
import openai

openai.api_key = "open_api_key"

# Load the pre-trained DETR model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")

# Set up the Tkinter window
window = tk.Tk()
window.title("Recipe Detection")
window.geometry("500x500")

# Function to handle image upload and recipe generation
def generate_recipes():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            # Open and display the selected image
            image = Image.open(file_path)
            image = image.resize((400, 300))  # Resize for display purposes
            img_tk = ImageTk.PhotoImage(image)
            image_label.configure(image=img_tk)
            image_label.image = img_tk

            # Process the image with DETR model
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)

            # Convert outputs (bounding boxes and class logits) to COCO API
            # Let's only keep detections with score > 0.7
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]

            detected_objects = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                detected_objects.append(model.config.id2label[label.item()])

            # Generate recipe descriptions using OpenAI API
            prompt = f"Create a recipe from the edible items from this list: {detected_objects}"
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=0.7,
                max_tokens=100,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            recipe = response.choices[0].text.strip()

            # Display recipe
            recipe_text.delete(1.0, tk.END)  # Clear previous text
            recipe_text.insert(tk.END, recipe)

        except Exception as e:
            recipe_text.delete(1.0, tk.END)  # Clear previous text
            recipe_text.insert(tk.END, f"Error: {str(e)}")

# Image display label
image_label = tk.Label(window)
image_label.pack(pady=10)

# Upload button
upload_button = tk.Button(window, text="Upload Image", command=generate_recipes)
upload_button.pack(pady=10)

#camera
upload_image = tk.Button(window, text="Capture Image")
upload_image.pack(padx=10)
# Recipe text box
recipe_text = tk.Text(window, height=100, width=50)
recipe_text.pack(pady=10)

# Run the Tkinter event loop
window.mainloop()
