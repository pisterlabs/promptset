from openai import OpenAI
from dotenv import load_dotenv
import sys
import os
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

if len(sys.argv) ==2:
    image_path = sys.argv[1]
    response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is this dog or cat species? (The answer must be a dog or cat species) just give me the name of the species ",
          },
          {
            "type": "image_url",
            "image_url": {
              "url": image_path,
            },
          },
          
        ],
      }
    ],
    max_tokens=600,
  )
    print(response.choices[0])