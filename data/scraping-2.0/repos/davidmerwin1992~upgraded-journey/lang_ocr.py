import sys
import subprocess
import openai
from PIL import Image
import pytesseract
import time
import os
import os
import pytesseract
from PIL import Image
import openai
import subprocess

# Path to image folder
image_dir = '/path/to/images'  

# OpenAI API key
openai_api_key = 'your_key'

# Supported formats
formats = ['.jpg', '.png'] 

def ocr_image(filepath):
    """Run OCR on image and return text"""
    image = Image.open(filepath)
    text = pytesseract.image_to_string(image)
    return text

def clean_text(text):
    """Clean up text for filename"""
    text = text.strip().replace(' ', '_') 
    return text

def categorize(text):
    """Use GPT-3 to categorize text"""
    # Call OpenAI API
    ...

def rename_file(filepath, text):
    filename = clean_text(text)[:50] + '.jpg'
    new_path = os.path.join(image_dir, filename)
    os.rename(filepath, new_path)

def add_metadata(filepath, text, category):
    """Add tags and comments""" 
    tags = category
    comments = text

    # Call xattr to set tags and comments
    ...

for filename in os.listdir(image_dir):

    filepath = os.path.join(image_dir, filename)

    if os.path.splitext(filename)[1].lower() in formats:

        text = ocr_image(filepath)

    category = categorize(text)

    rename_file(filepath, text) 

    add_metadata(filepath, text, category)
# Remove special characters, whitespace etc
new_name = cleanup_text(text) 

# Limit length
new_name = new_name[:50]
# Path to the folder where screenshots are saved
screenshot_folder = '/Users/david/Desktop/Screenshots_Automate'

for filename in os.listdir(image_dir):
    filepath = os.path.join(image_dir, filename)
    
# Supported image formats
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']

# OpenAI API key (replace with your actual key)
openai_api_key = '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'

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
    else:
        raise ValueError("Unsupported image format")

    return image

@handle_errors
def get_text_with_ocr(image):
    """Use Tesseract OCR to extract text from an image."""
    text = pytesseract.image_to_string(image)

    if not text:
        raise Exception("No text found in the image")

    return text

@handle_errors
def categorize(text):
    """Use OpenAI GPT-3 to categorize text."""
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

def process_screenshot(file_path):
    # Load image
    image = load_image(file_path)
    if not image:
        return

    # Extract text using OCR
    text = get_text_with_ocr(image)
    if not text:
        return

    # Classify text
    category = categorize(text)
    if not category:
        return
    text = get_text_with_ocr(filepath)
    # Set tags and comments
    set_tags_and_comments(file_path, category, "Your comments here")

    set_tags_and_comments(filepath, text, "Comments")
def monitor_folder():
    while True:
        for file_name in os.listdir(screenshot_folder):
            file_path = os.path.join(screenshot_folder, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(tuple(SUPPORTED_FORMATS)):
                process_screenshot(file_path)
                os.rename(filepath, os.path.join(image_dir, new_name))
                #os.remove(file_path)  #Optionally, delete the processed screenshot
        time.sleep(1)  # Adjust the sleep time as needed

if __name__ == "__main__":
    monitor_folder()
