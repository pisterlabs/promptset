
import os
import base64
import requests
from openai import OpenAI

class ImageTranslator:
    """
    A class to interact with OpenAI's GPT-4 Vision API, supporting single and multiple image analysis,
    with options for low and high fidelity image understanding.

    Attributes:
        api_key (str): Your OpenAI API key.
    """

    def __init__(self, api_key =None):
        """
        Initializes the ChatGPT Vision class with an API key.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

    def encode_image(self, image_path):
        """
        Encodes a local image to a base64 string.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def create_image_content(self, image_data, is_url=True, detail_level="auto"):
        """
        Prepares the image content for the payload.
        """
        if is_url:
            image_content = {"url": image_data, "detail": detail_level}
        else:
            # For base64, use the 'image_url' type with a data URI
            base64_data_uri = f"data:image/jpeg;base64,{image_data}"
            image_content = {"url": base64_data_uri, "detail": detail_level}

        return { "type": "image_url", "image_url": image_content }

    def analyze_images(self, images, max_tokens=300):
        """
        Analyzes single or multiple images using the GPT-4 Vision API.

        Parameters:
            images (list): A list of tuples (image_data, is_url, detail_level).
            max_tokens (int): The maximum number of tokens for the response.

        Returns:
            dict: The API response.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s in these images?"}
                ] + [self.create_image_content(*image) for image in images]
            }
        ]

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": messages,
            "max_tokens": max_tokens
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        return response.json()