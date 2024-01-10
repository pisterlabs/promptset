import os

import openai
import requests

# Set up your OpenAI API credentials
openai.api_key = open('apiKey.txt', 'r').read().strip()


def download_image(url):
    data = requests.get(url).content
    # Saving the image data to the local directory
    os.makedirs("localImgSave", exist_ok=True)
    file_path = os.path.join('localImgSave', 'Img.jpg')
    with open(file_path, 'wb') as file:
        file.write(data)


def genImage(varPrompt):
    response = openai.Image.create(
        prompt=varPrompt,
        n=1,
        # size="256x256"
        size="1024x1024"
    )
    imageUrl = response['data'][0]['url']
    print(imageUrl)
    download_image(imageUrl)
