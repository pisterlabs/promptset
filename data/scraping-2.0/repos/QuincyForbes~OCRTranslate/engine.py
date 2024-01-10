import logging
from PIL import Image
import cv2
import numpy as np
import google.cloud.vision as vision
from google.auth import load_credentials_from_file
import io
import os
import re
import openai
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

openai.api_key = config["OpenAPIKey"]

credentials, project = load_credentials_from_file("cred.json")
client = vision.ImageAnnotatorClient(credentials=credentials)

def preprocess_image(image):
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply dilation and erosion to remove some noise
        kernel = np.ones((1, 1), np.uint8)
        gray = cv2.dilate(gray, kernel, iterations=1)
        gray = cv2.erode(gray, kernel, iterations=1)

        # Apply threshold to get image with only black and white
        img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 8
        )

        return img
    except Exception as e:
        logging.error(f"An error occurred during image preprocessing: {e}")
        return None

def generate_text(prompt, lst):
    try:
        # Joining the text in the provided list
        prompt = prompt + str(lst)
        messages = [{"role": "user", "content": f"{prompt}"}]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
        )
        response_message = response["choices"][0]["message"]["content"]
        return response_message
    except Exception as e:
        logging.error(f"An error occurred during text generation: {e}")
        return None

def process_text(lst):
    try:
        cleaned_list = []
        for item in lst:
            item = re.sub(r"[\.\n]", "", item)  # Remove periods and newline characters
            item = re.sub(r"[0-9]", "", item)  # Remove numbers
            item = re.sub(r"[^\w\s]", "", item)  # Remove punctuation except whitespace
            item = re.sub(r"[a-zA-Z]", "", item)  # Remove English characters
            item = re.sub(
                r"\s+", " ", item
            )  # Replace multiple whitespaces with a single space
            item = item.strip()  # Remove leading/trailing spaces
            if item:
                cleaned_list.append(item)
        return cleaned_list
    except Exception as e:
        logging.error(f"An error occurred during text processing: {e}")
        return []

def process_chunk(chunk):
    try:
        # Convert image chunk to bytes
        is_success, buffer = cv2.imencode(".png", chunk)
        io_buf = io.BytesIO(buffer)
        byte_img = io_buf.read()

        image = vision.Image(content=byte_img)
        response = client.text_detection(image=image)
        texts = response.text_annotations

        # Draw the boxes on the original image
        for text in texts:
            vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
            cv2.polylines(chunk, [np.array(vertices)], True, (0, 0, 255), 2)

        if response.error.message:
            raise Exception(
                "{}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors".format(response.error.message)
            )

        return chunk, [text.description for text in texts]
    except Exception as e:
        logging.error(f"An error occurred during chunk processing: {e}")
        return None, []

def execute(image_path, prompt):
    try:
        img = Image.open(image_path)  # Load the image file
        logging.info(f"Image {image_path} loaded successfully.")
        img = np.array(img)

        chunks = []
        detected_texts = []

        # Set your chunk size (this is a square so x and y are the same)
        chunk_size = 3069

        # Calculate the number of chunks needed in the x and y directions
        num_x_chunks = img.shape[1] // chunk_size + (1 if img.shape[1] % chunk_size else 0)
        num_y_chunks = img.shape[0] // chunk_size + (1 if img.shape[0] % chunk_size else 0)

        for i in range(num_y_chunks):
            row_chunks = []
            for j in range(num_x_chunks):
                y_start = i * chunk_size
                y_end = (i + 1) * chunk_size

                x_start = j * chunk_size
                x_end = (j + 1) * chunk_size

                # This line extracts the chunk from the image
                chunk = img[y_start:y_end, x_start:x_end]
                processed_chunk, texts = process_chunk(preprocess_image(chunk))
                row_chunks.append(processed_chunk)
                detected_texts.extend(texts)
            chunks.append(row_chunks)

        # Reassembling the chunks
        reassembled_image = cv2.vconcat([cv2.hconcat(h_chunks) for h_chunks in chunks])

        processed_text = process_text(detected_texts)
        ai_complete = generate_text(prompt, processed_text)

        filename = os.path.basename(image_path)  # Get the base filename
        filename_without_extension = os.path.splitext(filename)[
            0
        ]  # Get the filename without extension
        proc_filename = f"proc_{filename_without_extension}.png"  # Processed chunk filename

        cv2.imwrite(os.path.join("outputs", proc_filename), reassembled_image)
        logging.info(f"Processed image saved as {os.path.join('outputs', proc_filename)}.")

        return proc_filename, ai_complete

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None, None
