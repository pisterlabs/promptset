import base64
import random
import textwrap

import aiohttp
import openai
import requests
from PIL.Image import Image

from PIL import Image, ImageDraw, ImageFont
from dotenv import dotenv_values
import io

async def a1_image_gen(prompts, parent_prompt_id=None):
    config = dotenv_values('.env')
    print("generating prompt: " + prompts[0].text)
    data = {
        "enable_hr": False if (prompts[0].parent_prompt or parent_prompt_id) else True,
        # "enable_hr": True,
        "denoising_strength": 0.45,
        "hr_scale": 2,
        "hr_second_pass_steps": 5,
        "hr_upscaler": "ESRGAN_4x",
        "prompt": "",
        "seed": prompts[0].seed,
        "n_iter": prompts[0].quantity,
        "steps": prompts[0].steps,
        "height": prompts[0].height,
        "width": prompts[0].width,
        "negative_prompt": prompts[0].negative_prompt,
        "sampler_name": config['SAMPLER'],
        "batch_size": int(config['BATCH_SIZE']),
        "cfg_scale": config["CFG_SCALE"],
        "script_name": "prompts from file or textbox",
        "script_args": [False, True, "\n".join([prompt.text for prompt in prompts])]
    }
    async def fetch(session, url):
        async with session.post(url, json=data, timeout=1000) as response:
            return await response.json()


    async with aiohttp.ClientSession() as session:
        r = await fetch(session, config['GRADIO_API_BASE_URL'] + 'sdapi/v1/txt2img')

    # r = requests.post(f"{config['GRADIO_API_BASE_URL']}sdapi/v1/txt2img", json=data)
    try:
        files = r["images"]
    except:
        print(r)
        return []
    return files


def dalle_image_gen(prompt):
    print("generating DALLE prompt: " + prompt.text)
    config = dotenv_values('.env')
    openai.api_key = config['OPENAI_API_KEY']
    try:
        images = openai.images.generate(
            model="dall-e-3",
            prompt=prompt.text,
            n=1,
            size="1024x1024",
            response_format="b64_json",
        )
    except:
        return [], prompt.text

    return [images.data[0].b64_json], images.data[0].revised_prompt


async def get_generation_from_api(prompts, parent_prompt_id=None) -> Image:
    images = []
    revised_prompt = ""
    # images = a1_image_gen(prompt)
    if prompts[0].method == "stable-diffusion":
        images = await a1_image_gen(prompts, parent_prompt_id)
    elif prompts[0].method == "dalle3":
        images, reworded_prompt = dalle_image_gen(prompts[0])
        revised_prompt = reworded_prompt
    return images, revised_prompt if prompts[0].method == "dalle3" else prompts[0].text


def calculate_font_size(caption):
    if len(caption) < 100:
        return 40
    elif len(caption) < 200:
        return 30
    return 20


def add_caption_to_image(img, caption, output_path):
    # Font settings

    font_size = calculate_font_size(caption)
    font_path = "arial.ttf"
    font = ImageFont.truetype(font_path, font_size)

    # Create a temporary drawing context to calculate text size
    temp_img = Image.new('RGB', (img.width, img.height), (255, 255, 255))
    temp_draw = ImageDraw.Draw(temp_img)

    # Calculate text size and wrap text
    margin = 20
    max_width = img.width - 2 * margin
    wrapped_text = textwrap.fill(caption, width=100)
    text_bbox = temp_draw.textbbox((0, 0), wrapped_text, font=font)
    wrap_count = 100
    while text_bbox[2] > max_width:
        wrap_count -= 1
        wrapped_text = textwrap.fill(caption, width=wrap_count)
        text_bbox = temp_draw.textbbox((0, 0), wrapped_text, font=font)

    # Calculate the height of the caption and create a new image with space for caption
    caption_height = text_bbox[3] + 2 * margin
    new_img = Image.new('RGB', (img.width, img.height + caption_height), (255, 255, 255))
    new_img.paste(img, (0, caption_height))

    # Draw the text onto the new image, centered
    draw = ImageDraw.Draw(new_img)
    text_width = text_bbox[2] - text_bbox[0]
    x = (new_img.width - text_width) // 2
    y = margin
    draw.text((x, y), wrapped_text, fill="black", font=font)

    # Save the new image
    return new_img


def batch_add_caption(generated_images, prompts):
    captioned_images = []

    for i, generated_image in enumerate(generated_images):
        try:
            text = prompts[i].text if len(prompts)>1 else prompts[0].text
        except IndexError:
            text = "something went wrong"
        captioned_images.append(add_caption_to_image(generated_image, text, "temp.png"))
    return captioned_images


async def batch_text_to_image(prompts, parent_prompt_id=None):
    generated_images, revised_prompt = await get_generation_from_api(prompts, parent_prompt_id)
    if len(generated_images) == 0:
        return [], ""
    # convert the b64 encoded images to PIL images
    generated_images = [Image.open(io.BytesIO(base64.b64decode(image))) for image in generated_images]

    captioned_images = batch_add_caption(generated_images, prompts)
    return captioned_images, revised_prompt

async def text_to_image(prompt, parent_prompt_id=None):
    generated_images, revised_prompt = await get_generation_from_api([prompt], parent_prompt_id)
    if len(generated_images) == 0:
        return [], ""
    # convert the b64 encoded images to PIL images
    generated_images = [Image.open(io.BytesIO(base64.b64decode(image))) for image in generated_images]

    captioned_images = batch_add_caption(generated_images, [prompt])
    return captioned_images, revised_prompt