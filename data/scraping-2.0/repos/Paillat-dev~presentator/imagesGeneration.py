import requests
import os
import openai
async def generate(prompt, path, mode, openai_key):
    #r = requests.get(f"http://localhost:8000/generate_image?prompt={prompt}&path={path}")
    if mode == "sd": 
        r = requests.get(f"http://localhost:8000/generate_image?prompt={prompt}&path={path}")
        return "image generated"
    if mode == "dalle":
        openai.api_key = openai_key
        img = await openai.Image.acreate(
                prompt=prompt,
                n=1,
                size="512x512",
            )
        return img