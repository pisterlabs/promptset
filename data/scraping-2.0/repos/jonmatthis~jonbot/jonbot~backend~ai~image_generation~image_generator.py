import asyncio
import os
import random
import string
import time
from io import BytesIO
from pathlib import Path
from typing import Optional, Literal

import openai
import requests
from PIL import Image
from dotenv import load_dotenv
from openai import AsyncOpenAI

from jonbot import logger
from jonbot.system.path_getters import get_base_data_folder_path


class ImageGenerator:
    def __init__(self):
        self.latest_image_path = None
        self.image_save_path = Path(get_base_data_folder_path()) / "generated_images"
        load_dotenv()
        # self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def generate_image(self,
                             prompt: str,
                             size: Literal["1024x1024", "1792x1024", "1024x1792"] = "1024x1024",
                             quality: Literal["standard", "hd"] = "standard",
                             style: Literal["vivid", "natural"] = "vivid",
                             n: int = 1) -> str:
        while True:
            try:
                response = await self.client.images.generate(model="dall-e-3",
                                                             prompt=prompt,
                                                             size=size,
                                                             quality=quality,
                                                             style=style,
                                                             n=n)
                break
            except openai.BadRequestError as e:
                if e.code == "content_policy_violation":
                    time.sleep(1)  # sleep it for a sec to make sure we don't send too many naughty requests in a row
                    logger.warning(
                        f"Prompt raised a 'content policy violation' error when generating an image with prompt: \n\n `{prompt}`\n\n - adjusting prompt and trying again")
                    completion = await self.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system",
                             "content": "This prompt raised a 'content policy violation' error when generating an image. "
                                        "Please minimally edit the prompt to remove content that may violate OpenAI's "
                                        "content policy (remove political, violent, sexual themes, etc). Make only minor changes, you will have another chance to try again "
                                        "if it gets rejected again"},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=len(prompt)
                    )
                    prompt = completion.choices[0].message.content
                    logger.info(f"Trying again with prompt: \n\n {prompt}\n\n")

        filename = await self.get_file_name_from_prompt(prompt)
        self.download_and_save_image(url=response.data[0].url, filename=filename)

        if not Path(self.latest_image_path).exists():
            raise FileNotFoundError(f"Image was not saved to {self.latest_image_path}")
        return str(self.latest_image_path)

    async def edit_image(self,
                         base_image_path: str,
                         prompt: str,
                         mask_path: Optional[str] = None,
                         n: int = 1,
                         size: Literal["1024x1024", "1792x1024", "1024x1792"] = "1024x1024") -> str:
        temp_file_name = "temp.png"
        og_filename = Path(base_image_path).name
        if mask_path is None:
            # convert image to RGBA and save to temp file
            img = Image.open(base_image_path)
            img = img.convert("RGBA")
            tmp_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
            tmp_img.paste(img)

            tmp_img.save(temp_file_name)
            base_image_path = temp_file_name
        else:
            raise NotImplementedError("Masked image editing is not yet implemented")

        with open(base_image_path, "rb") as image_file:
            response = await self.client.images.edit(image=image_file,
                                                     prompt=prompt,
                                                     n=n,
                                                     size=size

                                                     )
        if Path(temp_file_name).exists():
            os.remove(temp_file_name)

        base_filename = og_filename.split(".")[0] + "_edit"
        filename = base_filename + ".png"
        file_name_iteration = 0
        while Path(filename).exists():
            file_name_iteration += 1
            filename = base_filename + f"_{file_name_iteration}.png"

        image_path = self.download_and_save_image(response.data[0].url, filename)
        return image_path

    async def create_image_variation(self,
                                     base_image_path: str,
                                     n: int = 1,
                                     size: Literal["1024x1024", "1792x1024", "1024x1792"] = "1024x1024") -> str:
        with open(base_image_path, "rb") as image_file:
            response = await self.client.images.create_variation(
                image=image_file,
                n=n,
                size=size
            )
        og_filename = Path(base_image_path).name
        base_filename = og_filename.split(".")[0] + "_variation"
        filename = base_filename + ".png"
        file_name_iteration = 0
        while Path(filename).exists():
            file_name_iteration += 1
            filename = base_filename + f"_{file_name_iteration}.png"
        image_path = self.download_and_save_image(response.data[0].url, filename)

        return image_path

    def download_and_save_image(self, url: str, filename: str) -> Path:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        self.image_save_path.mkdir(parents=True, exist_ok=True)  # creates directory if it doesn't exist
        self.latest_image_path = self.image_save_path / filename
        img.save(self.latest_image_path)
        return self.latest_image_path

    async def get_file_name_from_prompt(self, prompt: str) -> str:
        logger.info(f"Generating filename from prompt: {prompt}")
        completion = await self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "Generate a highly condensed but descriptive filename (formated in `snake_case`) based on this prompt"},
                {"role": "user", "content": prompt},
            ]
        )
        filename = "generated_image"
        if completion.choices[0].message.content != "":
            filename = completion.choices[0].message.content.split(".")[0]
            if len(filename) > 50:
                filename = filename[:50]
            filename = filename.replace(" ", "_")
            filename = filename.replace("\n", "")
            filename = filename.replace(":", "")

        # generate random 6 digit hex string
        random_hex_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        filename += "_" + random_hex_string + ".png"
        logger.info(f"Generated filename: {filename}")
        return filename

    def display_image(self):
        if os.name == 'nt':
            os.system(f"start {str(self.latest_image_path)}")
        else:
            os.system(f"xdg-open {str(self.latest_image_path)}")


if __name__ == '__main__':
    image_generator = ImageGenerator()
    image_path = asyncio.run(
        image_generator.generate_image(
            prompt="an otherworldly entity, madness to behold (but otherwise kuwaii af) - put it on the ground in a forest"))
    image_generator.display_image()

    variation_image_path = asyncio.run(
        image_generator.create_image_variation(base_image_path=image_path))
    image_generator.display_image()

    edited_image_path = asyncio.run(
        image_generator.edit_image(base_image_path=image_path,
                                   prompt=" add a bunch of kittens running all over the place (and some dinosaurs)"))

    image_generator.display_image()
    print("done!")
