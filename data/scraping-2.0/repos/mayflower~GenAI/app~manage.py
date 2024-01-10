import base64
import debugpy
from flask import Flask
from flask.cli import FlaskGroup
import time
import os
from openai import OpenAI
from pathlib import Path
import requests

def create_app():
    app = Flask(__name__)
    return app

cli = FlaskGroup(create_app=create_app)

@cli.command('generate_on_image')
def generate_product_description_on_product_image():
  # Get the start time
  st = time.time()
  """
  debugpy.listen(("0.0.0.0", 5678))
  print("Waiting for client to attach...")
  debugpy.wait_for_client()
  """
  # Function to encode the image
  def encode_image(image_path):
    with open(image_path, "rb") as image_file:
      return base64.b64encode(image_file.read()).decode('utf-8')

  # Path to your image
  image_path = str(Path(os.environ.get("IMAGES_PATH")) / 'running_shoe.png')

  # Getting the base64 string
  base64_image = encode_image(image_path)

  headers = {
    "Content-Type": "application/json",
    "Authorization": f"""Bearer {os.environ.get("OPENAI_API_KEY")}"""
  }

  payload = {
    "model": os.environ.get("OPENAI_GPT_VISION_VERSION"),
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Create a product description for the image."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
            }
          }
        ]
      }
    ],
    "max_tokens": 300
  }

  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
  # Get the end time
  et = time.time()
  # Get the execution time
  elapsed_time = et - st
  print(response.json()['choices'][0]['message']['content'].strip())
  print('Execution time:', elapsed_time, 'seconds')

@cli.command('generate_similar')
def generate_product_description():
  # Get the start time
  st = time.time()
  client = OpenAI(
      api_key=os.environ.get("OPENAI_API_KEY")
  )

  response = client.chat.completions.create(
     messages=[
        {
            "role": "user",
            "content": """The product is a wireless Bluetooth earbuds with high definition sound quality, 20 hours of battery life, and a compact charging case.
            
            Describe a similar product:""",
        }
    ],
    model=os.environ.get("OPENAI_GPT_VERSION"),
    temperature=0.5,
    max_tokens=100,
  )
  # Get the end time
  et = time.time()
  # Get the execution time
  elapsed_time = et - st
  print(response.choices[0].message.content.strip())
  print('Execution time:', elapsed_time, 'seconds')

@cli.command('translate_eng_to_ger')
def translate_english_to_german():
  # Get the start time
  st = time.time()
  client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY") # Replace with your OpenAI API key
  )

  # Provide some instructions for the model
  instructions = """Translate the following English product description to German:
  
  English: This high-quality camera offers a resolution of 20 megapixels and a 5x optical zoom.
  
  German:"""
  
  # Generate the translation using ChatGPT
  response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": instructions,
        }
    ],
    model=os.environ.get("OPENAI_GPT_VERSION"),
    temperature=0.7,
    max_tokens=100,
  )

  german_translation = response.choices[0].message.content.strip()
  # Get the end time
  et = time.time()
  # Get the execution time
  elapsed_time = et - st
  print("German translation:", german_translation)
  print('Execution time:', elapsed_time, 'seconds')

if __name__ == "__main__":
  cli()