import json
import os
import openai
from base64 import b64decode
# create.py

def generate_img(prompt):
    PROMPT = prompt
    DATA_DIR = "/Users/emmanuellandau/Documents/generated_images"
    openai.api_key = os.environ["OPENAI_API_KEY"]

    response = openai.Image.create(
        prompt=PROMPT,
        n=1,
        size="1024x1024",
        response_format="b64_json",
    )

    file_name = os.path.join(DATA_DIR, f"{PROMPT[:5]}-{response['created']}.json")

    with open(file_name, mode="w", encoding="utf-8") as file:
        json.dump(response, file)




    # convert
    # ATA_DIR = Path.cwd() / "responses"
    JSON_FILE = file_name
    IMAGE_DIR = DATA_DIR



    with open(JSON_FILE, mode="r", encoding="utf-8") as file:
       response = json.load(file)

    for index, image_dict in enumerate(response["data"]):
        image_data = b64decode(image_dict["b64_json"])

        image_file = os.path.join(DATA_DIR, f"{PROMPT[:5]}-{response['created']}.png")
        with open(image_file, mode="wb") as png:
            png.write(image_data)
    return image_file