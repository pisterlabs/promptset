from openai import OpenAI
import base64
from dotenv import load_dotenv
import os
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# # image url
# response = client.chat.completions.create(
#     model="gpt-4-vision-preview",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": "Describe what the meme image is trying to convey"},
#                 {"type": "image_url", "image_url": { "url": "https://i.imgur.com/tskMYRf.jpeg"} },
#             ]
#         }
#     ],
#     max_tokens=300,
# )
# print(response.model_dump_json())

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = encode_image("ai-meme.jpg")

# local image
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe what the meme image is trying to convey"},
                {"type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{base64_image}"} },
            ]
        }
    ],
    max_tokens=300,
)

print(response.model_dump_json())

