import base64
import requests
import dotenv
import json
import os
import time
import logging
import datetime
from playsound import playsound
from openai import OpenAI

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("main_app.log"),
        logging.StreamHandler()
    ]
  )

IMAGE_PATH = "./media/CURRENT.jpg"
PROMPT = "You're task is to create a narration to this image. What is the person doing? What are their intentions? You can speculate freely."

# Elevenlabs configs

VOICE_ID = "7ocq291fMmdXFngd68ml"

dotenv.load_dotenv()
API_KEY_OPENAI = os.getenv("API_OPENAI")
client = OpenAI(
  api_key=API_KEY_OPENAI
)
API_KEY_ELEVENLABS = os.getenv("API_ELEVENLABS")
if API_KEY_OPENAI is None or API_KEY_ELEVENLABS is None:
  raise Exception("Api keys not set or env file is not loaded")
# OpenAI API Key


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Getting the base64 strin
def get_description(image_path = IMAGE_PATH):
  base64_image = encode_image(image_path)


  response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
      {
        "role": "user",
        "content": [
          {"type": "text", "text": PROMPT},
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}",
            },
          },
        ],
      }
    ],
    max_tokens=100,
  )


  timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_")
  filename = f"./media/descriptions/description_{timestamp}.txt"
  
  
  with open(filename, 'w') as f:
    f.write(response.choices[0].message.content)
  
  

  return response.choices[0].message.content



def call_elevenlabs_api(text, api_key=API_KEY_ELEVENLABS):
  CHUNK_SIZE = 1024
  url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"

  headers = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": api_key
  }

  data = {
    "text": text,
    "model_id": "eleven_monolingual_v1",
    "voice_settings": {
      "stability": 0.5,
      "similarity_boost": 0.5
    }
  }

  response = requests.post(url, json=data, headers=headers)
  timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_")
  filename = f"./media/audio/{timestamp}.mp3"
  with open(filename, 'wb') as f:
      for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
          if chunk:
              f.write(chunk)



  return filename
  


def main():
  interval = 5
  loop = 0
  max_loop = 10
  while True:
    logger.info("Getting description")
    description = get_description()
    logger.info("Calling Elevenlabs API")
    path_to_sound = call_elevenlabs_api(description)
    logger.info("Playing sound")
    playsound(path_to_sound)
    logger.info(f"Sleeping for {interval} seconds")
    time.sleep(interval)
    loop += 1
    if loop > max_loop:
      break
    


  
if __name__ == "__main__":
  main()




    
