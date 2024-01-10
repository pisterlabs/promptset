import huggingface_hub
import discord
import openai
import os
import logging
import torch
from torch import autocast
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionPipeline
from datasets import list_datasets
from dotenv import load_dotenv


log = logging.DEBUG
# print(log)

load_dotenv()


key = os.getenv('FRANKI_NO_ANIKI')


pipe = StableDiffusionPipeline.from_pretrained(
    'hakurei/waifu-diffusion',
    torch_dtype=torch.float32
).to('cuda')

prompt = "1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt"
with autocast("cuda") :
    image = pipe(prompt, guidance_scale=6)["sample"][0]

image.save("test.png")



# model_id = "prompth/openjourney"
# model_id_2 = "CompVis/stable-diffusion-v1-4"
# device = "cuda"
# pipeline = DiffusionPipeline.from_pretrained(model_id,)
# #pipe = StableDiffusionKDiffusionPipeline.from_pretrained(model_id)
# # pipe = pipe.to(device)
# prompt = "a photograph of an astronaut riding a horse"
# image = pipeline(prompt).images[0]
# image.save(f"retro_card.png")
# resultat= pipeline(prompt)
# print(resultat)
#
#
#
#




# prompt = input("Prompt: ")
# response = openai.Image.create(
#     prompt=prompt,
#     n=1,
#     size="1024x1024"
# )
#
# image_url = response['data'][0]['url']
#
#
# print(image_url)
#



















