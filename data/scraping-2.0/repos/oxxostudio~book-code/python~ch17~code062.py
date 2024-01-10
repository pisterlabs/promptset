# Copyright © https://steam.oxxostudio.tw

import openai
openai.api_key = '你的 API KEY'

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  max_tokens=128,
  temperature=0.5,
  messages=[
        {"role": "user", "content": "我叫做 oxxo"},
        {"role": "assistant", "content": "原來你是 oxxo 呀"},
        {"role": "user", "content": "請問我叫什麼名字？"}
    ]
)
print(response.choices[0].message.content)

