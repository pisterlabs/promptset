import os
import openai
from dotenv.main import load_dotenv

load_dotenv()
openai.api_key = os.environ['openai_api']
openai.api_base='https://202.79.168.46/v1'

# 单轮对话调用
# model可选"gpt-3.5-turbo"与"gpt-3.5-turbo-0301"
def generate_answer(messages):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7
    )
    res_msg = completion.choices[0].message
    return res_msg["content"].strip()


if __name__ == '__main__':
    # 维护一个列表用于存储多轮对话的信息
    messages = [{"role": "system", "content": "你现在是很有用的助手！"}]
    while True:
        prompt = input("请输入你的问题:")
        messages.append({"role": "user", "content": prompt})
        res_msg = generate_answer(messages)
        messages.append({"role": "assistant", "content": res_msg})
        print(res_msg)