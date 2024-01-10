# https://www.makeuseof.com/generate-images-using-openai-api-dalle-python/

import openai
import os
import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

class ImageGenerator:
    def __init__(self) -> str:
        self.image_url: str
        openai.api_key = "sk-XXX"
        self.APIKey = openai.api_key
        self.name = None

    def generateImage(self, Prompt, ImageCount, ImageSize):
        try:
            self.APIKey
            response = openai.Image.create(
                prompt=Prompt,
                n=ImageCount,
                size=ImageSize,
            )
            self.image_url = response["data"]

            self.image_url = [image["url"] for image in self.image_url]
            print(self.image_url)
            return self.image_url
        except openai.error.OpenAIError as e:
            print(e.http_status)
            print(e.error)

    def downloadImage(self, names) -> None:
        try:
            self.name = names
            for url in self.image_url:
                image = requests.get(url)
            for name in self.name:
                with open("{}.png".format(name), "wb") as f:
                    f.write(image.content)
        except:
            print("An error occured")
            return self.name

    def convertImage(self, maskName):
        image = Image.open("{}.png".format(maskName))
        rgba_image = image.convert("RGBA")
        rgba_image.save("{}.png".format(maskName))
        return rgba_image

    def editImage(self, imageName, maskName, ImageCount, ImageSize, Prompt) -> str:
        self.convertImage(maskName)
        response = openai.Image.create_edit(
            image=open("{}.png".format(imageName), "rb"),
            mask=open("{}.png".format(maskName), "rb"),
            prompt=Prompt,
            n=ImageCount,
            size=ImageSize,
        )
        self.image_url = response["data"]
        self.image_url = [image["url"] for image in self.image_url]
        print(self.image_url)
        return self.image_url


# Instantiate the class
imageGen = ImageGenerator()

# Edit an existing image:
imageGen.editImage(
    imageName="anim5",
    maskName="anim7",
    ImageCount=1,
    ImageSize="1024x1024",
    Prompt="An eagle standing on the river bank drinking water with a big mountain",
)

# Download the edited image:
imageGen.downloadImage(
    names=[
        "New Animals",
    ]
)
