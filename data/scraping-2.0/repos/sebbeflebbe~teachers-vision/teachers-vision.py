import os
import base64
from pdf2image import convert_from_path
import requests
import openai

# Function to convert PDF to high-resolution images
def convert_pdf_to_images(pdf_path, dpi=300):
    try:
        return convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []

# Function to encode image to base64
def encode_image_to_base64(image):
    try:
        from io import BytesIO
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image to base64: {e}")
        return None

# Function to analyze image with OpenAI
def analyze_image(base64_image, api_key):
    if not base64_image:
        return "No image provided"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",  # Verify if this model name is correct
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hur många rätta svar fanns på provet? Returnera enbart besvarade frågor. Lägg märke till att besvarade frågor kommer att vara skrivna med en penna."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 200
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return f"Error from OpenAI API: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error sending request to OpenAI API: {e}"

# Function to process all PDFs in a folder
def process_folder(folder_path, api_key):
    summaries = []
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return summaries

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Processing file: {pdf_path}")

            images = convert_pdf_to_images(pdf_path)
            for image in images:
                base64_image = encode_image_to_base64(image)
                result = analyze_image(base64_image, api_key)
                summaries.append(result)
    return summaries

# Main Execution
folder_path = './testfolder'  # Replace with the path to your folder containing PDFs
openai.api_key = ''  # Replace with your API key
all_summaries = process_folder(folder_path, openai.api_key)

# Print or process the summaries as needed
for summary in all_summaries:
    print(summary)
