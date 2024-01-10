import openai
import os

os.environ["all_proxy"] = 'http://127.0.0.1:10792' # 如果需要，请替换为你的代理地址

openai.api_key = "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"

# 创建一个包含系统消息和用户消息的对话
conversation = [
    {"role": "system", "content": "你是一位精通机器学习和自然语言处理的AI领域专家"},
    {"role": "user", "content": "我是一个小白，我想入门AI领域，我需要学习哪些知识"}
]

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k",
    messages=conversation
)

# 打印助手的回复
print(completion.choices[0].message)
