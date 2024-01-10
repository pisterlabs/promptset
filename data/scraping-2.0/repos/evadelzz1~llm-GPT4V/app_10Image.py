from dotenv import load_dotenv
from openai import OpenAI
import time

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)


# Initialize the OpenAI client with the API key
client = OpenAI()
# OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=OPENAI_API_KEY)

# https://platform.openai.com/docs/guides/vision
response = client.chat.completions.create(
  model="gpt-4-vision-preview",
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What’s in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
            "detail": "low"
          },
        },
      ],
    }
  ],
  max_tokens=300,
)

# print(response, "\n\n")
# print(response.choices[0], "\n\n")
print(response.choices[0].message.content, "\n\n")
print(response.usage, "\n\n")
print(response.usage.total_tokens)
# https://openai.com/pricing
