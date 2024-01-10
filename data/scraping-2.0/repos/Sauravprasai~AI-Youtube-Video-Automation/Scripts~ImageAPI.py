from openai import OpenAI
import time, requests, re

import requests

API_URL = "https://api-inference.huggingface.co/models/playgroundai/playground-v2-1024px-aesthetic"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content


def download_images_ai():
    with open('Text Folder/Images_name.txt', 'r') as file:
        cleaned_names = file.readlines()

    Images_names = [re.sub(r'[^a-zA-Z0-9\s]', '', name) for name in cleaned_names]

    for i, Images_names in enumerate(Images_names):
        images_name = Images_names.strip()
    
        response = query({
        "inputs": images_name,
        })

        with open(f'Images/{images_name}.jpg', 'wb') as f:
            f.write(response)

print("Images from AI downloaded successfully!")

