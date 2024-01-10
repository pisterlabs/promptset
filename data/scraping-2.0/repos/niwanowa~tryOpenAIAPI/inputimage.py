from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPEN_AI_KEY"),
)

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "この画像はどのようなプロンプトで生成可能ですか？"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://hugo.niwanowa.tips/twittericon.png",
                    },
                },
            ],
        }
    ],
    max_tokens=300,
)

print(response.model_dump_json(indent=2))