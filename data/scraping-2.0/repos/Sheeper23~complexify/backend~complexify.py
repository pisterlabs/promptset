import os
import csv
from PIL import Image
from fractions import Fraction
from transformers import BlipProcessor, BlipForConditionalGeneration
from config import OPENAI_API_KEY
import json

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to get dimensions of an image
def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.width, img.height

# Function to compute the aspect ratio of an image
def compute_aspect_ratio(width, height):
    return Fraction(width, height).limit_denominator()

# Function to get file size in MB
def get_file_size_mb(file_path):
    return os.path.getsize(file_path) / (1024 * 1024)

# Function to write a CSV file
def write_csv(csv_file, csv_data):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Type", "Number of Frames", "Height in Pixels", "Width in Pixels", "File Size in MB", "Description"])
        writer.writerows(csv_data)

# Function to get image caption using BLIP
def get_image_caption(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Main function
def main(image_path):
    # Prepare data for the CSV
    csv_data = []

    file_size_mb = get_file_size_mb(image_path)
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        width, height = get_image_dimensions(image_path)
        description = get_image_caption(image_path)
        frames = 1
        csv_data.append([os.path.basename(image_path), 'Image', frames, height, width, file_size_mb, description])

    # Write the CSV file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(current_dir, 'output.csv')
    write_csv(csv_file, csv_data)


# Run the main function
if __name__ == "__main__":
    main()