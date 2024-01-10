from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPEN_AI_KEY"),
)

response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
        {"role": "system", "content": "新興宗教コンサルティングのスペシャリストです。"},
        {"role": "user", "content": "ITエンジニアのための宗教を作ろうと思います。テックなネタに絡めたユーモラスな宗教の名前と教義を考えてください。また、レスポンスはJSON形式で返してください。"},
    ],
    response_format={ "type": "json_object" },
)

print(response.model_dump_json(indent=2))