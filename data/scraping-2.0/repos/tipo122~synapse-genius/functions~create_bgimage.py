import logging
import json
import os

import openai
import requests
from openai.error import APIConnectionError, APIError, RateLimitError, ServiceUnavailableError

from firebase_functions import https_fn

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def main(req: https_fn.Request) -> https_fn.Response:

    prompt = "beach, blue sky, for background image"
    image = generate_image_url_dalle(prompt)
 
    return json.dumps({"data" : image})

def generate_image_url_dalle(image_caption, dimensions=(512, 512)):
    try:
        image_response = openai.Image.create(
            prompt=(image_caption[:1000]),
            n=1,
            size=f"{dimensions[0]}x{dimensions[1]}",
            response_format="url",
        )
        log.debug(f"Image Response: {image_response}")
        url_image = image_response["data"][0]["url"]
    except openai.InvalidRequestError as e:
        log.warn(
            f"Skipping image generation. Image prompt was rejected by OpenAI: {e.args}")
        return None
    except (APIConnectionError, APIError, RateLimitError, ServiceUnavailableError) as e:
        log.warn(f"Temporary API error. Skipping image generation: {e.args}")
        return None
    else:
        return url_image
    
def generate_image_base64_dalle(image_caption, dimensions=(512, 512)):
    try:
        image_response = openai.Image.create(
            prompt=(image_caption[:1000]),
            n=1,
            size=f"{dimensions[0]}x{dimensions[1]}",
            response_format="b64_json",
        )
        log.debug(f"Image Response: {image_response}")
        base64_image = image_response["data"][0]["b64_json"]
    except openai.InvalidRequestError as e:
        log.warn(
            f"Skipping image generation. Image prompt was rejected by OpenAI: {e.args}")
        return None
    except (APIConnectionError, APIError, RateLimitError, ServiceUnavailableError) as e:
        log.warn(f"Temporary API error. Skipping image generation: {e.args}")
        return None
    else:
        return base64_image


def generate_image_base64_stability(image_caption, dimensions=(512, 512)):
    STABILITY_ENGINE_ID = "stable-diffusion-512-v2-1"  # "stable-diffusion-v1-5"
    # The latter arg is a default
    STABILITY_API_HOST = os.getenv("API_HOST", "https://api.stability.ai")
    STABILITY_API_KEY = os.environ["STABILITY_API_KEY"]

    print(f"Generating image with caption: {image_caption[:1000]}")

    response = requests.post(
        f"{STABILITY_API_HOST}/v1/generation/{STABILITY_ENGINE_ID}/text-to-image",
        headers={"Content-Type": "application/json", "Accept": "application/json",
                 "Authorization": f"Bearer {STABILITY_API_KEY}"},
        json={
            "text_prompts": [{"text": image_caption[:1000]}],
            "cfg_scale": 7,
            "clip_guidance_preset": "FAST_BLUE",
            "height": dimensions[1],
            "width": dimensions[1],
            "samples": 1,
            "steps": 50,
        },
    )
    if response.status_code != 200:
        log.warn(
            f"Skipping image generation. Stability response was {response.status_code}: {response.text}")
        return None
    else:
        data = response.json()
        return data["artifacts"][0]["base64"]


