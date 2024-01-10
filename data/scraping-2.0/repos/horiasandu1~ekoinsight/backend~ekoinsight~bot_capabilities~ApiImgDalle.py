from base64 import b64decode
import os
from .ApiOpenAi import ApiOpenAi
from ..templates import *
import openai

class ApiImgOpenAi(ApiOpenAi):
    def __init__(self,dry_run=False):
        self.dry_run=dry_run
        self.provider_name = "DALLE"
    def fetch_img(self,prompt,size="256x256"):
        """
        prompt "worn glass bottle lying in a mystical desert"
        size "256x256"
        """
        prompt+=" "+DALLE_positive_prompts+" ".join(DALLE_story_styles)
        
        if not self.dry_run:
            openai.api_key = self.key
            response=openai.Image.create(
                prompt=prompt,
                n=1,
                size=size,
                response_format="b64_json"
            )

            img=b64decode(response['data'][0].b64_json)

            file_name = f"{generated_images_folder}/" f"DALL_{prompt[:200]}-{response['created']}.png"

            with open(file_name, mode="wb") as png:
                png.write(img)

            return file_name

        else:
            print(f"would have created the image with {self.provider_name} using prompt {prompt}")
