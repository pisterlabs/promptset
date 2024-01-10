import base64
import json
import requests
import numpy as np
import cv2
# from real_gemini.gpt4v import GPT4V
import os
from openai import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
URL='http://192.168.77.1:8000/main/'
OPEN_AI_SYSTEM_PROMPT = """the user is dictating with his or her camera on.
they are showing you things visually and giving you text prompts.
be very brief and concise.
be extremely concise. this is very important for my career. do not ramble.
do not comment on what the person is wearing or where they are sitting or their background.
focus on their gestures and the question they ask you.
do not mention that there are a sequence of pictures. focus only on the image or the images necessary to answer the question.
don't comment if they are smiling. don't comment if they are frowning. just focus on what they're asking.
"""

def img2base64(imgs):
    base64_imgs = []
    if isinstance(imgs,list):
        for img in imgs:
            # with open(img,'rb') as f:
            if isinstance(img,str):
                with open(img,'rb') as f:
                    img_bytes = f.read()
            else:
                img_bytes = np.array(cv2.imencode('.png', img)[1]).tobytes()
            img_b64 = base64.b64encode(img_bytes).decode()
            base64_imgs.append(img_b64)
        return base64_imgs
    else:
        # with open(imgs,'rb') as f:
        if isinstance(imgs,str):
            with open(imgs,'rb') as f:
                img_bytes = f.read()
        else:
            img_bytes = np.array(cv2.imencode('.png', img)[1]).tobytes()
        img_bytes = np.array(cv2.imencode('.png', img)[1]).tobytes()
        img_b64 = base64.b64encode(img_bytes).decode()
        base64_imgs.append(img_b64)
        return base64_imgs

def gpt4v(query,imgs=None):
    imgs = img2base64(imgs)
    input_data = {
        'query':query,
        'base64_images':imgs
    }
    api_url = 'http://192.168.77.1:8000/main/'
    resp = requests.post(
        api_url, 
        headers={
            'Content-Type':'application/x-www-form-urlencoded',
            'accept':'application/json'
        },
        data=input_data, 
        )
    resp_data = resp.json()
    prompt_text = resp_data["response"]
    send = {
        'text':prompt_text
    }
    return send

def gpt4v_client(query,imgs=None):
    imgs = img2base64(imgs)
    current_file_list = []
    for base64_image in imgs:
        current_file_list.append(f"data:image/jpeg;base64,{base64_image}")

    messages = [
        {
            "role": "system",
            "content": OPEN_AI_SYSTEM_PROMPT,
            },
        ]
    
    content = []
    content.append({"type": "text", "text": query}) # query
    for image in current_file_list:
        content.append({"type": "image_url", "image_url": {"url": image}}) # images
    # print("len:",len(content))
    messages.append({"role": "user", "content": content}) # role
    try:
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=messages,
            max_tokens=256,
        )
    except Exception as e:
        print(e)
        return {'text':'response failed'}
    return {'text':response.choices[0].message.content}

if __name__ == '__main__':
    r = gpt4v(query='第二张图和第一张图片有什么内容？',imgs=['../source/1702803312680.png','../source/1702803359034.png'])
    print(r)