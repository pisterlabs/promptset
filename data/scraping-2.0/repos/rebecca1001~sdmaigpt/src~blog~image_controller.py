import time
from typing import List
from fastapi import HTTPException
from src.config import Config
import openai
import requests

leonardo_api_key = Config.LEONARDO_API_KEY
leonardo_model_id = Config.LEONARDO_MODEL_ID

class ImageController:
    def __init__(
        self,
        title: str,
        headers: List[str],
        width_of_image: int,
        height_of_image: int,
    ) -> None:
        self.title = title
        self.headers = headers
        self.width_of_image = width_of_image
        self.height_of_image = height_of_image

    def call_openai(self, prompt, model="gpt-4", temperature=0.8):
        """
        It is used to call openai and get response from it
        """

        try:
            response = openai.ChatCompletion.create(
                model=model, messages=prompt, temperature=temperature
            )

            # Extract desired output from JSON object
            content = response["choices"][0]["message"]["content"]
        except:
            try:
                response = openai.ChatCompletion.create(model=model, messages=prompt)

                # Extract desired output from JSON object
                content = response["choices"][0]["message"]["content"]
            except:
                raise HTTPException(detail="GPT error", status_code=400)

        return content

    def image_prompt_generation(self):
        messages = [
            {
                "role": "system",
                "content": "Act as a small DALLE-2 image prompt generator for Blogs. Image prompts are the one line prompts with the details like object, angle, style and all. Make sure generated image is realistic. The image generation can focus on one thing at a time so make sure that only one scenario or object is focused. You will be provided with the title and resolution and your task is to generate brief and enhanced prompt that can guide the image generator to generate the image. Include specifications like realistic, high quality, etc. Do not mention the respective outline before the prompt. At the end of the prompt add keywords for the detail like --realistic --HD --8k --..... Do not include anything related to blog in the prompt. Follow example for better image generation prompt. For example,\nTitle: Photo of a beautiful girl wearing casual shirt.\nPrompt: photo of a beautiful girl wearing casual shirt with a hoodie and leggings, city street, messy medium hair, slim body, view from back, medium upper body shot, looking at the camera, cute smile, shallow depth of field."
            },
            {
                "role": "user",
                "content": f"Title: {self.title}",
            },
        ]

        return self.call_openai(messages)

    def headings_image_prompt_generation(self, heading):
        messages = [
            {
                "role": "system",
                "content": "Act as a small DALLE-2 image prompt generator for Blogs. Image prompts are the one line prompts with the details like object, angle, style and all. Make sure that generated image is realistic. The image generation can focus on one thing at a time so make sure that only one scenario or object is focused. You will be provided with one outlines and its sub outlines. Generate only one image prompt for it. Do not mention the respective outline before the prompt. At the end of the prompt add keywords for the detail like --realistic --HD --8k --..... Do not include anything related to blog in the prompt. Follow example for better image generation prompt. For example,\nHeading: Photo of a handsome african american adult man.\nPrompt: photo of a handsome african american adult man, laying in bed, relaxes recline pose, resembles actor Jordan Calloway, buzz cut hair sides faded, athletic body, view from front, medium close up shot, looking directly into the camera, intense look, hazel eyes, almond colored skin, shallow depth of field.\nHeading: Create an image of a old and beautiful bridge from the front, and on the bridge, Must have several people.\nPrompt: Create an image of a old and beautiful bridge from the front, and on the bridge, Must have several people. they are talking on the bridge and they are walking front and i can see their faces, sky is Blue and sun is shining. Birds, happy people going to the town.",
            },
            {
                "role": "user",
                "content": f"Heading: {heading}",
            },
        ]

        return self.call_openai(messages)

    def generate_feature_image(self, prompt):
        url = "https://cloud.leonardo.ai/api/rest/v1/generations"

        payload = {
            "prompt": prompt,
            "negative_prompt": "cartoon, 2d, sketch, drawing, anime, open mouth, nudity, naked, nsfw, helmet, head gear, close up, blurry eyes, two heads, two faces, plastic, Deformed, blurry, bad anatomy, bad eyes, crossed eyes, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, mutated hands and fingers, out of frame, blender, doll, cropped, low-res, close-up, poorly-drawn face, out of frame double, blurred, ugly, too many fingers, deformed, repetitive, duplicate, black and white, grainy, extra limbs, High pass filter, airbrush, zoomed, soft light, deformed, extra fingers, mutated hands, missing legs, bad proportions , blind, bad eyes, ugly eyes, dead eyes, vignette, out of shot, gaussian, closeup, monochrome, grainy, noisy, text, writing, watermark, logo,over saturation, over shadow, [bad-hands-5] [worst quality:2], [low quality:2], [normal quality:2], low contrast",
            "modelId": leonardo_model_id,
            "width": self.width_of_image,
            "height": self.height_of_image,
            "num_images": 1,
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {leonardo_api_key}",
        }
        response = requests.post(url, json=payload, headers=headers)

        image_id = response.json()

        image_url = image_id["sdGenerationJob"]["generationId"]

        url = f"https://cloud.leonardo.ai/api/rest/v1/generations/{image_url}"

        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {leonardo_api_key}",
        }

        response = requests.get(url, headers=headers)
        output = response.json()
        image_link = image_url
        for _ in range(3):
            if output["generations_by_pk"]["status"] == "COMPLETE":
                image_link = output["generations_by_pk"]["generated_images"][0]["url"]
                break
            else:
                time.sleep(5)
                response = requests.get(url, headers=headers)
                output = response.json()

        return image_link

    def generate_images(self):
        try:
            main_image_url = ""
            if self.title:
                main_image_prompt = self.image_prompt_generation()
                main_image_url = self.generate_feature_image(main_image_prompt)

            urls_for_heading = []
            if self.headers:
                headings = self.headers

                for heading in headings:
                    headings_prompt = self.headings_image_prompt_generation(heading)
                    urls_for_heading.append(
                        self.generate_feature_image(headings_prompt)
                    )
        except:
            main_image_url = ""
            urls_for_heading = []
        return main_image_url, urls_for_heading
