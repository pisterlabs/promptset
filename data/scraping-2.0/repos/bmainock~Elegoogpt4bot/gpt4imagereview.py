from openai import OpenAI
import openai

def get_image_analysis(prompt,image_url):
  client = OpenAI(api_key = 'gpt4 api key')

  response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
      {
        "role": "user",
        "content": [
          {"type": "text", "text": prompt},
          {
            "type": "image_url",
            "image_url": {
              "url": image_url,
            },
          },
        ],
      }
    ],
    max_tokens=300,
  )

  print(response.choices[0])
  return response.choices[0].message.content

def get_image_analysis2(prompt,image_url):
  import base64
  import requests
  response = requests.get(image_url)
  with open("download path /memory_image.jpg", "wb") as file:
    file.write(response.content)
  # OpenAI API Key
  api_key = 'apikey'
  # Path to your image
  # image_path = 'http://127.0.0.1:5000/image/memory_image.jpg'
  image_path = "image path memory_image.jpg"
  # Function to encode the image
  def encode_image(image_path):
    with open(image_path, "rb") as image_file:
      return base64.b64encode(image_file.read()).decode('utf-8')



  # Getting the base64 string
  base64_image = encode_image(image_path)

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
            "text": prompt,
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
    "max_tokens": 3000
  }

  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
  print(response.json()['choices'][0]['message']['content'])
  return response.json()['choices'][0]['message']['content']
