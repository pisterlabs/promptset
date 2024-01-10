from PIL import Image
from openai import OpenAI
import base64
import json

def read_png(filepath):
    try:
        img = Image.open(filepath)
        return img
    except:
        print(f"Error: Cannot read file {filepath}")
        return None

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string

def upload_image_to_openai(image_path, llm_client: OpenAI):
    file = llm_client.files.create(
        file=open(image_path, 'rb'),
        purpose='assistants',
    )
    return file

def load_json(str: str):
    try:
        str = str.strip()
        if str.startswith("```json"):
            str = str.strip("```json")
            str = str.strip("```")
        print("Loading json str: ", str)
        return json.loads(str)
    except:
        return None
