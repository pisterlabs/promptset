from openai import OpenAI
import os
import requests
from pathlib import Path
from datetime import datetime

client = OpenAI()

def generate_and_save_images(descriptions, base_directory="generated_images", max_retries=3):
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("Error: OpenAI API key not found in environment variables.")
        return []

    today = datetime.now().strftime("%Y-%m-%d")
    save_directory = os.path.join(base_directory, today)
    Path(save_directory).mkdir(parents=True, exist_ok=True)

    saved_image_paths = []

    for i, description in enumerate(descriptions):
        retries = 0
        while retries < max_retries:
            try:
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=description,
                    n=1,
                    size="1024x1024"
                )

                if response.data:
                    image_url = response.data[0].url
                    image_response = requests.get(image_url)

                    if image_response.status_code == 200:
                        # Version control implementation
                        version = 1
                        file_name = f"image_{description}_{i}_v{version}.png"
                        file_path = os.path.join(save_directory, file_name)
                        while Path(file_path).exists():
                            version += 1
                            file_name = f"image_{description}_{i}_v{version}.png"
                            file_path = os.path.join(save_directory, file_name)

                        with open(file_path, 'wb') as file:
                            file.write(image_response.content)
                        saved_image_paths.append(file_path)
                        break
                    else:
                        print(f"Failed to download the image from {image_url}")

            except Exception as e:
                if "content_policy_violation" in str(e):
                    print(f"Content policy violation detected, retrying with a modified prompt. Attempt {retries + 1}")
                else:
                    print(f"An error occurred while generating or saving the image: {e}")
                    break
            retries += 1

    return saved_image_paths
