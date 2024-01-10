import os
import re
import openai
from dotenv import load_dotenv
import requests
from common import next_story_directory


def dalle_call(prompt, file_path):
    load_dotenv()
    openai.api_key = "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"
    response = openai.Image.create(prompt=prompt, n=1, size="1024x1024")
    image_url = response["data"][0]["url"]
    r = requests.get(image_url)
    if r.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(r.content)  # Write the contents (bytes) to a file


def generate_image(story):
    current_directory = os.getcwd()
    IMAGE_DIRECTORY = current_directory + "/images"
    CUR_STORY_DIR, _ = next_story_directory(base_dir=IMAGE_DIRECTORY, name="story")
    for idx, script in enumerate(story, start=1):
        image = script["image"]
        IMAGE_PATH = os.path.join(CUR_STORY_DIR, f"image{idx}.png")
        # dalle_call(image, IMAGE_PATH)
        get_unsplash_image(image, IMAGE_PATH)
        print(f"image{idx}.png created")


import requests
import shutil


def get_unsplash_image(search_query, filename):
    access_key = "YOUR_UNSPLASH_ACCESS_KEY"
    url = "https://api.unsplash.com/search/photos"

    headers = {"Authorization": "Client-ID {}".format(access_key)}

    querystring = {"query": search_query, "per_page": 1, "orientation": "landscape"}

    response = requests.request("GET", url, headers=headers, params=querystring)

    if response.status_code == 200:
        data = response.json()
        if data["results"]:
            image_url = data["results"][0]["urls"]["regular"]

            # Download the image and save it to a file
            image_response = requests.get(image_url, stream=True)
            if image_response.status_code == 200:
                with open(filename, "wb") as out_file:
                    shutil.copyfileobj(image_response.raw, out_file)
                return filename
            else:
                print(
                    "Error occurred when downloading image: ",
                    image_response.status_code,
                )
                return None
        else:
            return None
    else:
        print("Error occurred: ", response.status_code)
        return None


# filename = get_unsplash_image("black hole", "blackhole.jpg")
# print(filename)
