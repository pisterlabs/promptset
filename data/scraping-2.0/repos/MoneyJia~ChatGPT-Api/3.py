import openai
import os

# openai_api_key = os.environ.get("openai_api_key") #从系统环境变量读取
# openai_api_base = os.environ.get('openai_api_url') #从系统环境变量读取

os.environ["all_proxy"] = 'http://127.0.0.1:10792' #对应代理地址或科学上

openai_api_key = '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' #输入api_key
openai_api_base = 'https://api.nextweb.fun/openai/v1'

openai.api_key = openai_api_key
openai.api_base = openai_api_base

messages = [{'role':'user', 'content':'你好，请自我介绍一下'}]

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k-0613",
    messages=messages,
    stream=True
)

for r in response:
    if r.choices[0].finish_reason is None:
        print(r.choices[0].delta.content, end="")
    else:
        break


# curl https://api.openai.com/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer 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" \
#   -d '{
#     "model": "gpt-3.5-turbo",
#     "messages": [
#       {
#         "role": "system",
#         "content": "You are a helpful assistant."
#       },
#       {
#         "role": "user",
#         "content": "Hello!"
#       }
#     ]
#   }'