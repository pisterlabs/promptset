
from openai import AzureOpenAI
import os
import requests
from PIL import Image
import json
    

def get_dalle_image(prompt_str, image_save_name=None):

    fname_DALLE_KEY = 'DALLE_TOKEN.txt' # TODO: change to your own API key
    if os.path.isfile(fname_DALLE_KEY):
        with open(fname_DALLE_KEY, 'r') as fh:
            AZURE_DALLE_API_KEY = fh.read()
    else:
        print('Error: AZURE_DALLE_API_KEY file not found')
    AZURE_OPENAI_ENDPOINT = 'https://rhaetian-poppy-sweden.openai.azure.com/'

    client = AzureOpenAI(
        api_version="2023-12-01-preview",  
        api_key=AZURE_DALLE_API_KEY, #os.environ["AZURE_OPENAI_API_KEY"],  
        azure_endpoint=AZURE_OPENAI_ENDPOINT # os.environ['AZURE_OPENAI_ENDPOINT']
    )

    result = client.images.generate(
        model="rhaetian-poppy-dalle3", # the name of your DALL-E 3 deployment
        prompt=prompt_str,
        n=1
    )

    json_response = json.loads(result.model_dump_json())


    # Retrieve the generated image
    image_url = json_response["data"][0]["url"]  # extract image URL from response
    generated_image = requests.get(image_url).content  # download the image

    if image_save_name is not None:
        if os.path.isfile(image_save_name):
            print(f'Error: {image_save_name} already exists. Will overwrite.')
        else:
            os.makedirs(os.path.dirname(image_save_name), exist_ok=True)

        
        with open(image_save_name, "wb") as image_file:
            image_file.write(generated_image)

    return generated_image