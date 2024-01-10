from diffusers import StableDiffusionPipeline
from uuid import uuid4 as uuid
from openai import OpenAI
import torch
import os

def _generate(prompt, save_path):
    EXT = ".jpg"

    model_id = "dreamlike-art/dreamlike-photoreal-2.0"

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    image = pipe(prompt).images[0]

    image.save(f'{save_path}.{EXT}')
    
    return save_path + f".{EXT}"

def generate(prompt):
    client = OpenAI(api_key="API KEY HERE") 

    response = client.images.generate(
        model="dall-e-2",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1
    )

    return response.data[0].url