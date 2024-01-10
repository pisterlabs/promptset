# https://platform.openai.com/docs/guides/vision

from typing import List
import requests
import json

from real_gemini.utils.image_stacker import scale_and_stack_images, load_image, save_or_show_image, image2base64
from openai import OpenAI
import os

OPEN_AI_SYSTEM_PROMPT = """the user is dictating with his or her camera on.
they are showing you things visually and giving you text prompts.
be very brief and concise.
be extremely concise. this is very important for my career. do not ramble.
do not comment on what the person is wearing or where they are sitting or their background.
focus on their gestures and the question they ask you.
do not mention that there are a sequence of pictures. focus only on the image or the images necessary to answer the question.
don't comment if they are smiling. don't comment if they are frowning. just focus on what they're asking.
"""
# 从环境变量中获取API密钥
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

class GPT4V():
    def __init__(self) -> None:
        pass

    def process_stack_image_paths(
        self,
        query: str,
        image_paths: List[str],
        output_path: str = "./test/outputs/stacked_image.jpg"
    ):
        images = load_images(image_paths)
        stacked_image_base64 = scale_and_stack_images(images)
        # 保存或展示处理后的图像
        save_or_show_image(stacked_image_base64, output_path) # 保存到文件

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
            }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": query,
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{stacked_image_base64}"
                    }
                    }
                ]
                }
            ],
            "max_tokens": 300
            }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        # print(response.json())
        return response.json()
    
    def process_multi_image_base64(
        self,
        query: str,
        base64_images: List[str],
        output_path: str = "./test/outputs/request.json"
    ):

        current_file_list = []
        for base64_image in base64_images:
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
        with open(output_path, "w") as f:
            json.dump(messages, f, ensure_ascii=False, indent=4)

        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=messages,
            max_tokens=256,
        )

        return {"response": response.choices[0].message.content}

if __name__=="__main__":
    query = "Guess what movie I'm acting out."
    image_paths = ["./test/images/test_0.png", "./test/images/test_1.png", "./test/images/test_2.png", "./test/images/test_3.png"]
    output_path = "./test/outputs/stacked_image.jpg"
    print(query, image_paths)

    gpt4v = GPT4V()
    # result = gpt4v.process_stack_image_paths(query=query, image_paths=image_paths)

    base64_images = [image2base64(load_image(image_path)) for image_path in image_paths]
    result = gpt4v.process_multi_image_base64(query=query, base64_images=base64_images)

    print(result)