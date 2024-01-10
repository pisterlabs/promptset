import aiohttp
import requests
import openai
import json
import os
import urllib.request
import asyncio

from PIL import Image
from aiofiles import open as aio_open
from tqdm import tqdm

# Initialize OpenAI API
openai.api_key = "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"

async def generate_text(name, max_retries=3):
    retries = 0

    while retries < max_retries:
        try:
            messages = [
                {"role": "system", "content": """
                You are a helpful assistant that generates a detailed description for a single Pokémon. 
                Generate a thorough and comprehensive description. 
                The generated description should be in one paragraph.
                """},
                {"role": "user", "content": f"Generate a detailed description for the Pokémon: {name}."}
            ]

            # Increased max_tokens to 700
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, max_tokens=700)
            generated_text = response['choices'][0]['message']['content'].strip()
            return generated_text

        except Exception as e:

            print(f"An error occurred: {e} (attempt {retries + 1}/{max_retries}). Retrying...")

        retries += 1

    print("Max retries reached. Could not generate text.")

async def main():
    print("Fetching Pokémon data from local folder...")
    source_folder = "pokemon_images"
    destination_folder = "train_first_gen_pokemon_dataset/image"

    os.makedirs(destination_folder, exist_ok=True)

    pokemon_names = [name for name in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, name))]
    pokemon_names.sort()  # Optional: Sort names

    annotations = []
    filter_cap_path = 'train_first_gen_pokemon_dataset/filter_cap.json'
    if os.path.exists(filter_cap_path):
        with open(filter_cap_path, 'r') as f:
            existing_data = json.load(f)
            annotations = existing_data.get('annotations', [])

    # init the count as the max image_id + 1
    count = max([int(ann['image_id']) for ann in annotations]) + 1 if annotations else 0

    for name in tqdm(pokemon_names[78:]):
        print(f"Generating a detailed description for {name}...")
        description = await generate_text(name)
        print("Description generated!")
        print(description)

        if description:
            image_files = [f for f in os.listdir(os.path.join(source_folder, name)) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            for image_file in image_files:
                random_id = f"{count}"
                src_path = os.path.join(source_folder, name, image_file)
                dest_path = os.path.join(destination_folder, f"{random_id}.jpg")

                count += 1

                with Image.open(src_path) as img:
                    img.convert('RGB').save(dest_path, "JPEG")  # Convert to JPEG

                annotations.append({
                    "image_id": random_id,
                    "caption": description
                })

                print("Saving annotations...")
                with open("train_first_gen_pokemon_dataset/filter_cap.json", 'w') as f:
                    json.dump({"annotations": annotations}, f, indent=4)
                print("Annotations saved!")

if __name__ == '__main__':
    asyncio.run(main())