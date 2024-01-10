from handyllm import OpenAIAPI
from handyllm import utils

import json
from dotenv import load_dotenv, find_dotenv
# load env parameters from file named .env
# API key is read from environment variable OPENAI_API_KEY
# organization is read from environment variable OPENAI_ORGANIZATION
load_dotenv(find_dotenv())

## or you can set these parameters in code
# OpenAIAPI.api_key = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
# OpenAIAPI.organization = None

response = OpenAIAPI.images_generations(
    prompt="A very cool panda",
    n=1,
    size="256x256",
    # timeout=10,
)
print(json.dumps(response, indent=2))
download_url = response['data'][0]['url']
file_path = utils.download_binary(download_url)
print(f"generated image: {file_path}")


mask_file_path = "mask.png"
with open(file_path, "rb") as file_bin:
    with open(mask_file_path, "rb") as mask_file_bin:
        response = OpenAIAPI.images_edits(
            image=file_bin,
            mask=mask_file_bin,
            prompt="A cute panda wearing a beret",
            n=1,
            size="512x512",
            # timeout=10,
        )
print(json.dumps(response, indent=2))
download_url = response['data'][0]['url']
file_path = utils.download_binary(download_url)
print(f"edited image: {file_path}")


with open(file_path, "rb") as file_bin:
    response = OpenAIAPI.images_variations(
        image=file_bin,
        n=1,
        size="512x512",
        # timeout=10,
    )
print(json.dumps(response, indent=2))
download_url = response['data'][0]['url']
file_path = utils.download_binary(download_url)
print(f"varied image: {file_path}")

