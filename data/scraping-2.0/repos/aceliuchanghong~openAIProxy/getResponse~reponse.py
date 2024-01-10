import os
import httpx
from openai import OpenAI

proxyHost = "127.0.0.1"
proxyPort = 10809

client = OpenAI(http_client=httpx.Client(proxies=f"http://{proxyHost}:{proxyPort}"))
client.api_key = os.getenv("OPENAI_API_KEY")
completion = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
        {"role": "user",
         "content": "你是哪个模型?给我你的模型编号,是gpt-4-1106-preview还是gpt-3.5,还是其他?"
         }
    ]
)
print(completion.choices[0].message.content)
print(completion.choices[0].message.content)
