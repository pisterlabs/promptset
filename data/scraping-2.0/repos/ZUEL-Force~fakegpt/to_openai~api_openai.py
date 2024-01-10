import os
from io import BytesIO
from pathlib import Path

import openai
import requests
from myTools import get_salt
from PIL import Image
from private import *

openai.api_key = API_KEY


def chatgpt(que: list):
    try:
        response = openai.ChatCompletion.create(model=MODEL[0], messages=que)
        text = response['choices'][0]['message']['content']
        # reason = response['choices'][0]['finish_reason']
        # cost = response['usage']['total_tokens']
        text = str(text).strip()
        return text, 0
    except:
        return "后端连接超时，请稍后再试。", 1


def chatgpt_stream(que: list):
    try:
        response = openai.ChatCompletion.create(model=MODEL[0],
                                                messages=que,
                                                stream=True)
        for it in response:
            for choice in it.choices:
                yield choice.delta.content if "content" in choice.delta else ""
    except:
        return "server error, please try again later."


#TODO:语音转文字
def whisper(file: str):
    try:
        audio_file = open(os.path.join(AUDIO_FOLDER, file), 'rb')
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        return transcript["text"], 0
    except:
        return '语音转文字失败，请稍后再试。', 1


#TODO:文字转图片
def dalle(prompt: str):
    try:
        response = openai.Image.create(prompt=prompt, n=1, size="512x512")
        image_url = response['data'][0]['url']

        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        img_name = get_salt(6) + '.jpg'
        img.save(f"{PROJECT_PATH}/{IMG_FOLDER}/{img_name}")

        image_url = f"{PROJECT_PATH}/{IMG_FOLDER}/{img_name}.jpg"
        image_url = Path.as_uri(Path(image_url))
        return f'{IMG_FOLDER}{img_name}', 0
    except:
        return '图像生成失败，请稍后再试。', 1


# if __name__ == '__main__':
#     image_url, state = dalle("在绿色的草坪上有一只白色的猫咪")
#     print(image_url)
