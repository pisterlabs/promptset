import json
import os
from pathlib import Path
import openai
from base64 import b64decode
from pathlib import Path
import glob

def main(input_prompt):
    if len(input_prompt) > 256:
        input_prompt = input_prompt[:256]
    DATA_DIR = Path.cwd() / "responses"
    DATA_DIR.mkdir(exist_ok=True)
    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.Image.create(
        prompt=input_prompt,
        n=1,
        size="256x256",
        response_format="b64_json",
    )
    file_name = DATA_DIR / f"{input_prompt[:5]}-{response['created']}.json"

    with open(file_name, mode="w", encoding="utf-8") as file:
        json.dump(response, file)

    DATA_DIR = Path.cwd() / "responses"

    json_files = glob.glob(str(DATA_DIR / "*.json"))

    # If there are any json files, use the latest one (assuming it's the right one)
    if json_files:
        JSON_FILE = Path(max(json_files, key=os.path.getctime))
    else:
        raise FileNotFoundError(
            "No JSON files found in directory that match the pattern 'An ec-'."
        )

    IMAGE_DIR = Path.cwd() / "images" / JSON_FILE.stem

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    with open(JSON_FILE, mode="r", encoding="utf-8") as file:
        response = json.load(file)

    for index, image_dict in enumerate(response["data"]):
        image_data = b64decode(image_dict["b64_json"])
        image_file = IMAGE_DIR / f"{JSON_FILE.stem}-{index}.png"
        with open(image_file, mode="wb") as png:
            png.write(image_data)
            print(f"Saved {image_file}.")
            return "image.jpg"
