from PIL import Image
import functools
import os
import glob
import openai
import json
import random
import argparse

CAPTION_PROMPT = """
Describe the celestial object as if it was a prompt for an image generation model for the surface texture.

- Do not include the name of the celestial object itself.
- Caption only, it should not be a command. At most 2 sentences.
- Be scientific, clear/exact, and not artistic. Note colors and high level features
- The descriptions should be specific, visual, and mostly geological. Use plain concise language

Examples: 
- "a moon heavily cratered with a dark grey surface, a large ridge across the center"
- "a planet with a large green ocean and a small brownish continent in the middle"

Generate 10 captions.
"""


def _generate_captions(name):
    functions = [
        {
            "name": "write_captions",
            "description": "Write captions",
            "parameters": {
                "type": "object",
                "properties": {"captions": {"type": "array", "items": {"type": "string"}}},
                "required": ["captions"],
            },
        }
    ]
    resp = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {
                "role": "system",
                "content": CAPTION_PROMPT,
            },
            {"role": "user", "content": name},
        ],
        functions=functions,
        function_call={"name": "write_captions"},
    )
    args = json.loads(resp.choices[0]["message"]["function_call"]["arguments"])
    print(name, "->", args["captions"])
    return args["captions"]


@functools.lru_cache(maxsize=1000)
def generate_captions(name, retries=3):
    try:
        return _generate_captions(name)
    except openai.error.OpenAIError as e:
        if retries > 0:
            return generate_captions(name, retries=retries - 1)
        else:
            raise e


def build_dataset(textures_path, images_path):
    i = 0
    with open(os.path.join(images_path, "metadata.jsonl"), "w") as metacsv:
        for fn in glob.glob(os.path.join(textures_path, "*.txt")):
            print(fn)
            id_ = os.path.splitext(os.path.basename(fn))[0]
            with open(fn, "r") as f:
                lines = f.readlines()
                title = lines[0].strip()
                img = Image.open(os.path.join(textures_path, f"{id_}.png"))
            try:
                caption = random.choice(generate_captions(title))
            except Exception as e:
                print(fn, title, e)
                continue
            img_fn = f"{i:04d}.png"
            img.convert("RGB").save(os.path.join(images_path, img_fn))
            meta = dict(file_name=img_fn, text=caption)
            metacsv.write(f"{json.dumps(meta)}\n")
            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--textures_path", type=str)
    args = parser.parse_args()

    images_path = os.path.join(args.dataset_path, "train", "textures")
    os.makedirs(images_path, exist_ok=True)
    build_dataset(args.textures_path, images_path)
