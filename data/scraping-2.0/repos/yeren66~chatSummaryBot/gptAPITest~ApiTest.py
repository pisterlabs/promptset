import os
import openai
from dotenv import load_dotenv
 
# 加载.env文件中的环境变量
load_dotenv()
 
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")

with open('chata.txt', 'r', encoding='utf-8') as file:
    content = file.read()

# print(content)

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[{"role": "user", "content": "请帮我总结以下的一段群聊对话:\n" + content}]
)

print(completion.choices[0].message.content)
