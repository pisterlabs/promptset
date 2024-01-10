from decouple import config
from openai import OpenAI

client = OpenAI(api_key=config("OPENAI_API_KEY"))


response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {
                    "type": "image_url",
                    "image_url": "https://i.imgur.com/M6bTG.jpeg",
                },
            ],
        }
    ],
    max_tokens=300,
)

print(response.choices[0])


# 1024 x 1024 square image in detail: high mode costs 765 tokens or $0.00765


# non-English languages, especially if they have a alphabets beyond latin, rotated images and graphs, exact location in space of objects, and things like counting.
