import openai
from django.http import JsonResponse

from .utils import get_apikey
import requests
import os
from django.conf import settings
from openai import OpenAI

# Set the API base URL and key (ensure these values are stored securely)

get_apikey(openai)
client = OpenAI()


def generate_system_message(user_inputs, results_dict_cls, exif_dict):
    system_message = ""
    categories = user_inputs['photo_category']
    category_names = [str(category) for category in categories]
    category_str = ', '.join(category_names)
    if user_inputs['platform'].name == '微信' or user_inputs['platform'].name == '小红书':
        system_message = f"""
        
        You will be act to share a {user_inputs['platform']} Moments and only use {user_inputs['language']} language 
        as output. \
        
        
        You should put more focus on {category_str} and here is the special requirement from client: 
        {user_inputs['special_request']}. \
        
        I will provide the result from yolov8 and exif_data and you can use for detect time and objects to assist you 
        generating the Moments.\
        
        yolov8n result are {str(results_dict_cls)}, select the most possible object to expand description, do not
        mentioned probability in output.  
        {str(results_dict_cls)} \
        
        exif_information are following {str(exif_dict)}, ONLY attention to DateTimeOriginal (Time) and ignore others \
        
        
        Refer to the format that most people send {user_inputs['platform']} moments and avoid jargon to output text
        """
        print(system_message)
    else:
        system_message = f"""
              
              You will be act as Professional photography reviewer for 1x.com and only use {user_inputs['language']} language as output.\ 
              You should put more focus on {category_str} and here is the special requirement from client
              {user_inputs['special_request']}. \

              I will provide the result from yolov8n-cls and exif_data to assist you generate the description for time and objects,
              these result are in dictionary format.\

              yolov8n-cls result as 'OBJECT':probability format, select the most possible to expand description,do not
              mentioned probability in output.  
              {str(results_dict_cls)} \

              exif_information are following, attention to Time which is described in terms of DateTimeOriginal and 
              describe in photographic terms based on other information. 
              {str(exif_dict)}\

              Refer to the format that most people send {user_inputs['platform']} and output text
              """

    message = get_completion_messages(system_message)
    return get_completion_from_messages(message)


def get_completion_messages(system_message):
    return [
        {
            'role': 'system',
            'content': system_message
        }
    ]


def get_completion_from_messages(
        messages,
        model="gpt-4-1106-preview",
        temperature=0.8,
        max_tokens=2000
):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


# 调用chatgpt 语音 API
def generate_corrected_transcript(temperature, audio_file, combined_request):
    if combined_request.strip():  # 使用strip()来确保不仅仅是空格
        system_prompt = "Please help me answer the user's questions, I will give you the user's previous questions " \
                        + combined_request
    else:
        system_prompt = "Please help me answer the user's questions."

    # 转录用户的语音输入
    user_transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_transcript.text
            }
        ]
    )

    try:
        gpt_response = response.choices[0].message.content
    except KeyError:
        gpt_response = "Error: Unexpected response structure from the API."
    # 返回用户的转录文本和GPT的响应
    return {
        'user_transcript': user_transcript.text,
        'gpt_response': gpt_response
    }


def generate_corrected_text(temperature, text_info, combined_request):
    if combined_request.strip():  # 使用strip()来确保不仅仅是空格
        system_prompt = "Please help me answer the user's questions, I will give you the user's previous questions " \
                        + combined_request
    else:
        system_prompt = "Please help me answer the user's questions."

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": text_info
            }
        ]
    )

    try:
        gpt_response = response.choices[0].message.content
    except KeyError:
        gpt_response = "Error: Unexpected response structure from the API."
    # 返回用户的转录文本和GPT的响应
    return {
        'user_transcript': text_info,
        'gpt_response': gpt_response
    }


def generate_image(prompt, username, size="1024x1024"):
    # 使用OpenAI API生成图像
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url

    # 从URL下载图像
    image_response = requests.get(image_url, stream=True)
    image_response.raise_for_status()

    save_path_directory = os.path.join(settings.BASE_DIR, 'media', 'image_generate')
    if not os.path.exists(save_path_directory):
        os.makedirs(save_path_directory)

    save_path = os.path.join(save_path_directory, f"{username}.png")
    return_path = os.path.join('/media', 'image_generate', f"{username}.png")
    # 保存图像到指定路径
    with open(save_path, 'wb') as file:
        for chunk in image_response.iter_content(chunk_size=8192):
            file.write(chunk)

    return return_path
