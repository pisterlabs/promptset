import pyutils.cli.flags as pyflags

import openai
import requests
import shutil
import os


openai.api_key = os.environ["CHATGPT_API_KEY"]


def save_dalle_image(prompt, file_name, size="256x256", n=1):
    image_resp = openai.Image.create(prompt=prompt, n=n, size=size)

    if n == 1:
        image_url = image_resp['data'][0]['url']
    else:
        # Choose the first image URL if you requested multiple images
        image_url = image_resp['choices'][0]['data'][0]['url']

    response = requests.get(image_url, stream=True)
    response.raise_for_status()

    with open(file_name, 'wb') as file:
        response.raw.decode_content = True
        shutil.copyfileobj(response.raw, file)

    print(f"Image saved as {file_name}")


if __name__ == '__main__':
    
    prompt = pyflags.get_flag(
        flag_name="--prompt", 
        default_value="two dogs playing chess, oil painting", 
        prompt_label="prompt"
    )
    file_name = pyflags.get_flag(
        flag_name="--output", 
        default_value="dogs_playing_chess.jpg", 
        prompt_label="file_name"
    )

    save_dalle_image(prompt, file_name)
