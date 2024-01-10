import sys
import os
import io
from google.cloud import vision
from PIL import Image
import cv2
import subprocess  # Added for setting metadata
import plistlib  # Added for reading metadata
import openai

# Supported image formats
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']

# Error handling decorator
def handle_errors(func):
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error: {e}")
            return None
    return inner

@handle_errors
def load_image(image_path):
    """Load an image based on its extension and return an image object."""
    ext = os.path.splitext(image_path)[1].lower()

    if ext in SUPPORTED_FORMATS:
        image = Image.open(image_path)
    elif ext == '.bmp':
        image = cv2.imread(image_path)
    else:
        raise ValueError("Unsupported image format")

    return image

@handle_errors
def get_text(image):
    """Use Google Cloud Vision API to extract text from an image."""
    client = vision.ImageAnnotatorClient()
    with io.open(image, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(f'Google API Error: {response.error.message}')

    return texts[0].description if texts else None

@handle_errors
def categorize(text):
    """Use OpenAI GPT-3 to categorize text."""
    openai_api_key = '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'  # Replace with your OpenAI API key
    openai.api_key = openai_api_key
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt="Categorize and tag this image description: " + text,
            max_tokens=50
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error in GPT-3 request: {e}")
        return None

def set_tags_and_comments(image_path, tags, comments):
    try:
        # Set tags using `xattr`
        subprocess.run(['xattr', '-w', 'com.apple.metadata:_kMDItemUserTags', tags, image_path])

        # Set comments using `xattr`
        subprocess.run(['xattr', '-w', 'com.apple.metadata:kMDItemFinderComment', comments, image_path])
    except Exception as e:
        print(f"Error setting tags and comments: {e}")

def main(image_path):
    # Load image
    image = load_image(image_path)
    if not image:
        return

    # Extract text 
    text = get_text(image)
    if not text:
        return

    # Classify text
    category = categorize(text)
    if not category:
        return

    print(category)

    # Set tags and comments
    set_tags_and_comments(image_path, category, "Your comments here")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_path>")
    else:
        main(sys.argv[1])
        