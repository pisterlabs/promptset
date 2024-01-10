import openai
import os
from openai import OpenAI
import pyperclip

# 目前需要设置代理才可以访问 api
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7891"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7891"

api_key = ''
with open('key', 'r') as file:
    api_key = file.readline().strip()

client = OpenAI(api_key=api_key)

def f1():
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Act as an academic expert with specialized knowledge in computer science."},
            {
                "role": "user",
                "content": user_input+'The text is as follows:'+text1,
            }
        ],
        model="gpt-3.5-turbo",
    )

    res = completion.choices[0].message.content
    # 在res中每隔两个句号加上换行符
    
# 以流式输出的方式获取结果
def chat(query):
    num = 0
    stream = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Act as an academic expert with specialized knowledge in computer science."},
            {
                "role": "user",
                "content": user_input+'The text is as follows:'+text1,
            }
        ],
        model="gpt-3.5-turbo",
        stream=True,
    )
    for chunk in stream:
        slice = chunk.choices[0].delta.content or ""
        if '。' in slice:
            num += 1
            if num == 2:
                # index = slice.find('。')
                # slice = slice[:index+1] + '---' + slice[index+1:]
                # print(slice, end="",flush=True)
                # num = 0
                print(slice+'\n\n', end="",flush=True)
                num = 0
            else:
                print(slice, end="",flush=True)
        else:
            print(slice, end="",flush=True)


f2()
