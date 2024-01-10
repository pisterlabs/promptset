import requests
from openai import OpenAI
from pathlib import Path
from typing import Optional

from openai.types import ImagesResponse


def call_dalle_and_save_image(prompt: str, client: OpenAI, output_file_path: Path) -> Optional[Path]:
    try:
        # Generate image using OpenAI client
        response: ImagesResponse = client.images.generate(
            prompt=prompt, n=1, model="dall-e-3", size="1024x1024", quality="hd", response_format="url"
        )

        # Extract the image URL
        image_url = response.data[0].url
        if not image_url:
            print("No image URL found in the response.")
            return None

        print(image_url)

        # Download the image
        image_response = requests.get(image_url)
        if image_response.status_code == 200:
            # Write the image data to a file
            with open(output_file_path, "wb") as file:
                file.write(image_response.content)
            return output_file_path
        else:
            print(f"Error downloading image: {image_response.status_code}")
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
