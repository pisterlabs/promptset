from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
  model="gpt-4-1106-preview",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "谁获得了2022年英雄联盟S赛世界冠军？"},
  ]
)
print(response)