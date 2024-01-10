import base64
import requests
import os
import json
from utils.decorators import log
from PIL import Image
import io

# OpenAI API Key
api_key = os.environ.get('OPENAI_API_KEY')

class Getparams():
  '''get params from image and text content.'''
  def __init__(self, user_prompt:str, img:Image) -> None:
    self.user_prompt = user_prompt
    self.img = img
    
  def encode_image(self) -> base64:
    # use byte io to save PIL Image
    buffered = io.BytesIO()
    self.img.save(buffered, format="JPEG")
    img_data = buffered.getvalue()
    img_base64_str = base64.b64encode(img_data).decode('utf-8')

    return img_base64_str

  @log
  def get_params(self, prompt:str, img:base64) -> dict:

    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
      "model": "gpt-4-vision-preview",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": prompt
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{img}"
              }
            }
          ]
        }
      ],
      "max_tokens": 300
    }

    try:
      response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    except Exception as req_e:
      print(f"Faild to request, {req_e}")

    # resolve
    try:
      params = response.json()['choices'][0]['message']['content'].replace("'",'"')
      params = json.loads(params)
    except Exception as res_e:
      print(f"Failed to resolve the response from openai, {res_e}\n")
    
    return params
  

if __name__ == '__main__':
  image_path = "./small_img_2.jpg"
  sys_prompt_path = "./config/sys_prompt.txt"
  user_prompt_path = "./config/user_prompt.txt"

  param4img = Getparams(user_prompt_path, image_path)

  params = param4img.get_params(param4img.combine_prompt(sys_prompt_path), param4img.encode_image())
 
  