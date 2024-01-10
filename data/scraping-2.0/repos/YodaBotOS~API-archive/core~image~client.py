import io
import uuid
import json
import random
import asyncio
import functools

import openai
import aiohttp
from PIL import Image as PILImage

from .enums import *
from .image import *
from .style import *


class GenerateArt:
    def __init__(self, s3, keys: tuple):
        self.openai_key = keys[0]
        openai.api_key = keys[0]
        self.dream_key = keys[1]
        self.computer_vision_keys, self.computer_vision_region = keys[2]

        self.s3, self.bucket, self.host = s3

    @property
    def style(self):
        return GenerateStyleArt((self.s3, self.bucket, self.host), self.dream_key)

    def _get_headers(self):
        return {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json"
        }

    async def _upload_to_cdn(self, gen: GeneratedImages):
        # Read the url in gen.images[].url and upload it to boto3 S3 client.

        counter = 1
        img_id = uuid.uuid4()

        for image in gen.images:
            original_url = image.url

            async with aiohttp.ClientSession() as sess:
                async with sess.get(original_url) as resp:
                    data = await resp.read()

            # Upload to S3
            self.s3.upload_fileobj(
                io.BytesIO(data),
                Bucket=self.bucket,
                Key=f"art/{img_id}/{counter}.png"
            )

            image.url = f"{self.host}/art/{img_id}/{counter}.png"

            counter += 1

    async def create_image(self, prompt: str, n: int, *, size: Size, user: str = None) -> GeneratedImages:
        if 1 > n or n > 10:
            raise ValueError("n must be between 1 and 10")

        s = size.get_size()

        response = openai.Image.create(
            prompt=prompt,
            n=n,
            size=s,
            user=user
        )

        gen = GeneratedImages(self, response)
        await self._upload_to_cdn(gen)

        return gen

    async def create_image_variations(self, image: str | bytes | io.BytesIO, n: int, *, size: Size,
                                      user: str = None) -> GeneratedImages:
        if 1 > n or n > 10:
            raise ValueError("n must be between 1 and 10")

        s = size.get_size()

        if isinstance(image, str):
            image = open(image, "rb").read()
        elif isinstance(image, io.BytesIO):
            image = image.getvalue()

        response = openai.Image.create_variation(
            image=image,
            n=n,
            size=s,
            user=user
        )

        gen = GeneratedImages(self, response)
        await self._upload_to_cdn(gen)

        return gen

    def _convert_img(self, image):
        img = PILImage.open(image)
        new_img = io.BytesIO()
        img.save(new_img, format="PNG")

        new_img.seek(0)
        return new_img

    async def analyze(self, image: str | bytes | io.BytesIO) -> dict:
        key = random.choice(self.computer_vision_keys)

        if isinstance(image, str):
            image = io.BytesIO(open(image, "rb").read())
        elif isinstance(image, bytes):
            image = io.BytesIO(image)

        args = functools.partial(self._convert_img, image)
        image = await asyncio.get_event_loop().run_in_executor(None, args)

        img_id = uuid.uuid4()

        self.s3.upload_fileobj(
            image,
            Bucket=self.bucket,
            Key=f"art/analyze/{img_id}.png"
        )

        url = f'https://{self.host}/art/analyze/{img_id}.png'

        async with aiohttp.ClientSession() as sess:
            async with sess.post(
                f"https://{self.computer_vision_region}.api.cognitive.microsoft.com/vision/v3.2/analyze",
                data=json.dumps({"url": url}),
                headers={
                    "Ocp-Apim-Subscription-Key": key,
                    "Content-Type": "application/json"
                },
                params={
                    "visualFeatures": ",".join(["Tags", "Objects", "Brands", "Description", "ImageType", "Color",
                                                "Adult"]),
                    "model-version": "latest",
                    "language": "en",
                }
            ) as resp:
                js = await resp.json()
                
                if resp.status == 400:
                    return {"error": {"code": 400, "message": js["error"]["message"]}}

                data = {
                    "adult": {
                        "isAdultContent": js["adult"]["isAdultContent"],
                        "isRacyContent": js["adult"]["isRacyContent"],
                        "isGoryContent": js["adult"]["isGoryContent"],
                        "adultScore": js["adult"]["adultScore"] * 100,
                        "racyScore": js["adult"]["racyScore"] * 100,
                        "goreScore": js["adult"]["goreScore"] * 100
                    },
                    "tags": [{"name": i["name"], "confidence": i["confidence"] * 100} for i in js["tags"]],
                    "captions": [{"text": i["text"], "confidence": i["confidence"] * 100} for i in js["description"]["captions"]],
                    "color": js["color"],
                    "imageType": {
                        "isClipArt": True if js["imageType"]["clipArtType"] >= 2 else False,
                        "isLineDrawing": True if js["imageType"]["lineDrawingType"] else False,
                        "clipArtType": js["imageType"]["clipArtType"],
                        "clipArtTypeDescribe": (
                            "non-clipart" if js["imageType"]["clipArtType"] == 0 else
                            "ambiguous" if js["imageType"]["clipArtType"] == 1 else
                            "normal-clipart" if js["imageType"]["clipArtType"] == 2 else
                            "good-clipart" if js["imageType"]["clipArtType"] == 3 else
                            "unknown"
                        )
                    },
                    "brands": [{"name": o["name"], "confidence": o["confidence"] * 100, 
                                "rectangle": o["rectangle"]} for o in js["brands"]],
                    "objects": [{"object": o["object"], "confidence": o["confidence"] * 100, 
                                 "rectangle": o["rectangle"]} for o in js["objects"]],
                    "metadata": {
                        "width": js["metadata"]["width"],
                        "height": js["metadata"]["height"]
                    }
                }

                return data
