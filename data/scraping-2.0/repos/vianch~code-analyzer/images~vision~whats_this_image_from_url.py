from openai import OpenAI
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


api_key=os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
  model="gpt-4-vision-preview",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What’s in this image?"},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://oaidalleapiprodscus.blob.core.windows.net/private/org-09FA30fCDeRvH0CKk92txDkz/user-GZha0W6pa4xXnyyk7PsH2P7Y/img-K2UCZXJ2QKGyGPxYsddH6Ql3.png?st=2023-11-20T19%3A06%3A13Z&se=2023-11-20T21%3A06%3A13Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-11-20T11%3A26%3A57Z&ske=2023-11-21T11%3A26%3A57Z&sks=b&skv=2021-08-06&sig=KLultYwrkx3mamLFXXGWBVyg1QA14Iv96aMKm/bnqbg%3D",
          },
        },
      ],
    }
  ],
  max_tokens=300,
)

print(response.choices[0])