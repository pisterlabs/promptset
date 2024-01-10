# Copyright © https://steam.oxxostudio.tw

import openai
openai.api_key = '你的 API Key'

response = openai.Completion.create(
    model="text-davinci-003",
    prompt="講個笑話來聽聽",
    max_tokens=128,
    temperature=0.5,
)

completed_text = response["choices"][0]["text"]
print(completed_text)

