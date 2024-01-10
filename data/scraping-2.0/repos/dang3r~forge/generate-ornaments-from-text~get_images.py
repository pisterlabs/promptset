import os
import pathlib
import requests
from openai import OpenAI
import concurrent.futures
import datetime

client = OpenAI()

# List of image prompts
objects = [
    "a belgian malinois",
    "a malinois",
    "a malinois from the side standing with their tail behind them"
]


# Define the number of" workers
num_workers = 3


def download_and_save_image(object: str, index: int) -> None:
    try:
        prompt = f"""A 3D model of a single (only one) `{object}` whose full body is visible from the front.
        Make sure all of the object is visible and in frame. Please only include the object and nothing else!
        Make sure the edges of the object are sharp and not too blurry.
        """
        timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
        norm_prompt = '_'.join(object.split()) 
        file_path = pathlib.Path(f"./images/{timestamp}-{norm_prompt}.png")

        print(f"Generating image for prompt: `{prompt}`")
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        print(f"Generated image for prompt: `{prompt}`")

        image_url = response.data[0].url
        image_data = requests.get(image_url).content
        with open(file_path, "wb") as f:
            f.write(image_data)
    except Exception as error:
        print(error)


# Create a ThreadPoolExecutor with the specified number of workers
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    for i, prompt in enumerate(objects):
        print(i, prompt)
        executor.submit(download_and_save_image, prompt, i)
