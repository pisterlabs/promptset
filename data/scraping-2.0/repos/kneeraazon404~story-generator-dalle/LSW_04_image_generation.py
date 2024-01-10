"""
This module provides functionality to generate images from text prompts using OpenAI's API
and post the results to a webhook.
"""

import time
import logging
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize the OpenAI client
client = OpenAI()


def generate_images_from_prompts(image_prompts):
    """
    Generates images for a given set of prompts and posts them to a webhook.
    Returns a dictionary of image URLs where the key is the prompt identifier.

    :param image_prompts: A dictionary of prompts where the key is a unique identifier for the prompt.
    :return: A dictionary of generated image URLs.
    """
    images = {}

    for prompt_key, prompt_text in image_prompts.items():
        logging.info(f"{prompt_key} is in progress...")
        attempt = 0
        max_attempts = 3

        while attempt < max_attempts:
            try:
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=prompt_text,
                    size="1024x1024",
                    quality="standard",
                    n=1,
                )
                image_url = response.data[0].url
                images[prompt_key] = image_url
                logging.info(f"Generated image URL for {prompt_key}: {image_url}")

                break
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed for {prompt_key}: {e}")
                attempt += 1
                time.sleep(1)

            if attempt == max_attempts:
                logging.error(
                    f"Failed to generate image for {prompt_key} after {max_attempts} attempts."
                )

    return images
