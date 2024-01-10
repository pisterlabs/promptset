import contextvars

from dotenv import load_dotenv
import openai
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

openai.proxy = "http://127.0.0.1:7890"
message = []
while True:
    d = input("User：")
    if d == "-1":
        break
    message.append({"role": "user", "content": d})
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=message
        )
    except openai.error.RateLimitError:
        print("访问过于频繁")
        continue
    message.append({"role": "assistant", "content": completion.choices[0].message.content})
    # 输出生成的内容
    print(completion.choices[0].message.content)
print("=============END==============")
