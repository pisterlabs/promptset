from openai import OpenAI

import dotenv
import os

# Load environment variables from .env file
dotenv.load_dotenv()

# Initialize the OpenAI client
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

response = client.chat.completions.create(
  model="gpt-4-vision-preview",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What is happening in this image? Is this an animated Gif? If yes, can you tell me what is going on with the animation?"},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://cdn.dribbble.com/users/1525393/screenshots/15227792/media/403b6662b658a44a82dd3554f4e63b83.gif",
          },
        },
      ],
    }
  ],
  max_tokens=300,
)

print(response.choices[0])