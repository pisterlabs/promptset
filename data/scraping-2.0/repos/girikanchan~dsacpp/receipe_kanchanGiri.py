import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import requests
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
import os
import openai

openai.api_key = "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"  # Replace with your actual OpenAI API key

# Load the pre-trained DETR model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")

# Set up the Tkinter window
window = tk.Tk()
window.title("Recipe Detection")
window.geometry("500x600")

# Function to handle image upload
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            # Open and display the selected image
            image = Image.open(file_path)

            # Resize the image while maintaining aspect ratio
            longest_edge = max(image.size)
            resized_size = 400
            if longest_edge > resized_size:
                aspect_ratio = resized_size / longest_edge
                new_width = int(image.size[0] * aspect_ratio)
                new_height = int(image.size[1] * aspect_ratio)
                image = image.resize((new_width, new_height))

            img_tk = ImageTk.PhotoImage(image)
            image_label.configure(image=img_tk)
            image_label.image = img_tk

            # Process the image with DETR model
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)

            # Convert outputs (bounding boxes and class logits) to COCO API
            # Let's only keep detections with score > 0.9
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]

            detected_objects = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                detected_objects.append(model.config.id2label[label.item()])

            # Generate recipe descriptions using OpenAI API
            recipe_prompt = f"Create a recipe from the edible items from this list: {', '.join(detected_objects)}"
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=recipe_prompt,
                temperature=1,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            # Clear previous recipe text
            recipe_text.delete(1.0, tk.END)

            # Display recipe descriptions
            recipe_text.insert(tk.END, response["choices"][0]["text"])

        except Exception as e:
            messagebox.showerror("Error", str(e))

# Image display label
image_label = tk.Label(window)
image_label.pack()

# Upload button
upload_button = tk.Button(window, text="Upload Image", command=upload_image)
upload_button.pack()

# Recipe text box
recipe_text = tk.Text(window, height=20, width=60)
recipe_text.pack()

# Run the Tkinter event loop
window.mainloop()
