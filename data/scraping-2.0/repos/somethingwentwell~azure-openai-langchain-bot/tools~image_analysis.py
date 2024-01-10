import os
from langchain.agents import Tool
from dotenv import load_dotenv
import aiohttp
import requests
import json

BASEDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
load_dotenv(os.path.join(BASEDIR, '.env'), override=True)

def describe_image(image):
        try:
            url = f"https://teccognitiveservices.cognitiveservices.azure.com/computervision/imageanalysis:analyze?api-version=2022-10-12-preview&features=description,tags"
            payload = json.dumps(
            {
                "url": image
            })

            headers = {
            'Ocp-Apim-Subscription-Key': "ec34e4ba9f484cbc9a78e48df96dd089",
            'Content-Type': 'application/json'
            }

            response = requests.post(url, headers=headers, data=payload)
            response_json = response.json()
            print(response_json)
            return str(response_json)
                    
        except Exception as e:
                print(f"Error: {e}")
                return f"Error: {e}"


async def async_describe_image(image):
        try:
            url = f"https://teccognitiveservices.cognitiveservices.azure.com/computervision/imageanalysis:analyze?api-version=2022-10-12-preview&features=description,tags"
            payload = json.dumps(
            {
                "url": image
            })

            headers = {
            'Ocp-Apim-Subscription-Key': "ec34e4ba9f484cbc9a78e48df96dd089",
            'Content-Type': 'application/json'
            }

            async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, data=payload) as response:
                            response_json = await response.json()
                            return str(response_json)

        except Exception as e:
                print(f"Error: {e}")
                return f"Error: {e}"
        
def ocr_image(image):
        try:
            url = f"https://teccognitiveservices.cognitiveservices.azure.com/computervision/imageanalysis:analyze?api-version=2022-10-12-preview&features=read"
            payload = json.dumps(
            {
                "url": image
            })

            headers = {
            'Ocp-Apim-Subscription-Key': "ec34e4ba9f484cbc9a78e48df96dd089",
            'Content-Type': 'application/json'
            }

            response = requests.post(url, headers=headers, data=payload)
            response_json = response.json()
            return str(response_json['readResult']['content'])
                    
        except Exception as e:
                print(f"Error: {e}")
                return f"Error: {e}"


async def async_ocr_image(image):
        try:
            url = f"https://teccognitiveservices.cognitiveservices.azure.com/computervision/imageanalysis:analyze?api-version=2022-10-12-preview&features=read"
            payload = json.dumps(
            {
                "url": image
            })

            headers = {
            'Ocp-Apim-Subscription-Key': "ec34e4ba9f484cbc9a78e48df96dd089",
            'Content-Type': 'application/json'
            }

            async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, data=payload) as response:
                            response_json = await response.json()
                            return str(response_json['readResult']['content'])

        except Exception as e:
                print(f"Error: {e}")
                return f"Error: {e}"

def ImageAnalysis():
    tools = []
    tools.append(Tool(
        name = "Azure Cognitive Services Vision Image Description",
        func=describe_image,
        description="Useful for when you need to analysis an image. Input must be a URL to the image.",
        coroutine=async_describe_image,
    ))
    return tools

def OCR():
    tools = []
    tools.append(Tool(
        name = "Azure Cognitive Services Vision OCR",
        func=ocr_image,
        description="Useful for when you need to OCR an image. Input must be a URL to the image.",
        coroutine=async_ocr_image,
    ))
    return tools

def image_analysis():
    tools = []
    tools.extend(ImageAnalysis())
    tools.extend(OCR())
    return tools