import base64
from openai import OpenAI

client = OpenAI()
MAX_TOKENS = 500

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_image(image_path, analysis_prompt):
    base64_image = encode_image(image_path)
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": analysis_prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{base64_image}",
                    },
                ],
            }
        ],
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content
