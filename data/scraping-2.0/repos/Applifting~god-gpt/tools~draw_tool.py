import asyncio
import json
import os
import uuid
from dotenv import load_dotenv
import httpx
from langchain.tools.base import BaseTool
import simpleaudio as sa
from speech_recognition import requests
import subprocess

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def open_image_non_blocking(path):
    subprocess.Popen(['open', path])



def download_image(url,image_name, output_folder="pics"):
    uuid_suffix = str(uuid.uuid4())[:6]
    image_path = os.path.join(output_folder, f"img_{uuid_suffix}.png")
    urlretrieve(url, image_path)
    print(f"Image saved at: {image_path}")
    return image_path



async def generate_image(prompt, api_key, output_folder="pics", n=1, size="512x512"):
    url = "https://api.openai.com/v1/images/generations"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    final_prompt = f"{prompt}, beautiful digital art, highly detailed, dream-like, 4k, unreal, ultra-detail"
    
    data = {
        "prompt": final_prompt,
        "n": n,
        "size": size,
        "response_format": "url"
    }

    print(f"Sending request to OpenAI API with data:\n{json.dumps(data, indent=2)}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=data, timeout=30.0)
            response.raise_for_status()
            response_data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while sending the request: {e}")
            if response.status_code != 200:
                print(f"Response from OpenAI API:\n{response.text}")
            return
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for i, img_data in enumerate(response_data['data']):
        img_url = img_data["url"]
        return await asyncio.to_thread(download_image,img_url, prompt, output_folder)






class DrawTool(BaseTool):
    name = "draw"
    description = "Use this tool you want to create or draw or make something. Describe to the tool what you want to create and it will create it for you."

    def _run(self, query: str, run_manager = None) -> str:
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(generate_image(query, openai_api_key))
        open_image_non_blocking(result)

        return f"Beautiful thing has been created. Make some witty comment about it."

    
    async def _arun(self, query: str, run_manager = None) -> str:
        raise NotImplementedError("custom_search does not support async")





