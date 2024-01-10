from fastapi import UploadFile
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
import os
import dotenv
import numpy as np
import cv2
import json
import base64
from langchain.schema.output_parser import StrOutputParser
from app.myTool import contentTool, chainTool, agentTool
dotenv.load_dotenv()

CACHE_DIRECTORY = "./app/cache"

def encode_image_to_base64(image: np.ndarray) -> str:
    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError("Could not encode image to JPEG format.")
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return encoded_image

def prompt_stt(audio_path: str) -> str:
    audio = open(audio_path, "rb")
    client = OpenAI()
    transcript = client.audio.transcriptions.create(
        file=audio,
        model="whisper-1", 
        language="ko",
        response_format="text"
    )
    return transcript

def prompt_image(image_path: str, prompt: str, location: str) -> str:
    image = cv2.imread(image_path)
    base64_image = encode_image_to_base64(image)

    client = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=300)
    human_message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
    )

    response = client.invoke([human_message])
    return response

def cache_file(content: UploadFile) -> str:
    os.makedirs(CACHE_DIRECTORY, exist_ok=True)
    file_path = os.path.join(CACHE_DIRECTORY, content.filename)
    with open(file_path, "wb") as file:
        file.write(content.file.read())
    return file_path

def prompt_chat(input: dict) -> str:
    data = {
        "system": input["input"]["location"],
        "input": input["input"]["text"]
    }
    agent = agentTool.agent_executor
    result = agent.invoke(data)
    return result["output"]