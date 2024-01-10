import sys
import logging
import multiprocessing
from PIL import Image
import pytesseract
import time
import os
import openai
import re
import subprocess
import tempfile
import shutil

# Path and configs
image_dir = '/path/images'
formats = ('.jpg', '.png')

logging.basicConfig(filename='app.log', level=logging.INFO)

def ocr_image(filepath):
    try:
        image = Image.open(filepath)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        logging.error(f"OCR error on {filepath}: {e}")
        return None

def clean_text(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with a single space
    text = text.strip().replace(' ', '_')
    return text

def categorize(text):
    openai.api_key = "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"  # Replace with your OpenAI API key
    try:
        response = openai.Completion.create(prompt=f"Categorize: {text}", ...)
        return response.choices[0].text
    except Exception as e:
        logging.error(f"Error in categorization request: {e}")
        return "Uncategorized"  # Default to Uncategorized in case of an error

def rename_file(filepath, text):
    new_name = f"{clean_text(text)}.jpg"
    new_path = os.path.join(image_dir, new_name)
    os.rename(filepath, new_path)
    return new_path

def add_metadata(filepath, text, category):
    tags = category
    comment = text
    try:
        subprocess.run(["xattr...", tags, comment, filepath])
    except Exception as e:
        logging.error(f"Error adding metadata: {e}")

def process_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in formats:
        return

    text = ocr_image(filepath)

    if not text:
        return

    category = categorize(text)

    new_filepath = rename_file(filepath, text)

    add_metadata(new_filepath, text, category)

def main():
    pool = multiprocessing.Pool()
    filepaths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]

    pool.map(process_file, filepaths)

if __name__ == '__main__':
    main()

# Testing
def test_clean_text():
    input_text = "   This is some text with extra spaces.   "
    cleaned_text = clean_text(input_text)
    assert cleaned_text.strip() == "This is some text with extra spaces", "Text cleaning failed."

def test_categorize():
    input_text = "Sample text for categorization"
    category = categorize(input_text)
    assert category != "Uncategorized", "Categorization failed."

def test_rename_file():
    fd, tmp = tempfile.mkstemp()
    filepath = f"/tmp/{os.path.basename(tmp)}"
    new_filepath = rename_file(filepath, "Test text")
    assert new_filepath == f"/path/images/{clean_text('Test text')}.jpg", "File renaming failed."
    shutil.rmtree(tempfile.gettempdir())

if __name__ == '__main__':
    # Run tests
    test_clean_text()
    test_categorize()
    test_rename_file()

    print("All tests passed.")
    