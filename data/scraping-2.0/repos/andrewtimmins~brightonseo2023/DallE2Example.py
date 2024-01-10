#
# BrightonSEO April 2023, Dall-E2 example.
#

import json
import os
from base64 import b64decode
from pathlib import Path

import openai

openai.api_key = "***key here***"

PROMPT = "Super mario on the beach"
DATA_DIR = Path.cwd() / "Dall-E"

DATA_DIR.mkdir(exist_ok=True)

response = openai.Image.create(
    prompt=PROMPT,
    n=1,
    size="1024x1024",
    response_format="b64_json",
)

file_name = DATA_DIR / f"{PROMPT[:5]}-{response['created']}.json"

with open(file_name, mode="w", encoding="utf-8") as file:
    json.dump(response, file)

with open(file_name, mode="r", encoding="utf-8") as file:
    response = json.load(file)

for index, image_dict in enumerate(response["data"]):
    image_data = b64decode(image_dict["b64_json"])
    image_file = DATA_DIR / f"{file_name.stem}-{index}.png"
    with open(image_file, mode="wb") as png:
        png.write(image_data)
        
