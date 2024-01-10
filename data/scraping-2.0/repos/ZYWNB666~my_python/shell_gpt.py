#!/bin/python3
import argparse
import json
import os

os.environ['OPENAI_API_KEY'] = 'sk-XN5z*********************Ho4Z'

from openai import ChatCompletion
import openai
import requests

# 初始化 ChatGPT
chat = ChatCompletion(engine="gpt-3.5-turbo")


# 自定义函数，用于与notebook_session插件进行交互
def call_notebook_session(code):
    url = "http://127.0.0.1:8000/api/run_cell"
    payload = {"code": code}
    response = requests.post(url, json=payload)
    return response.json()


def handle_response(response, messages):
    assistant_message_content = response['choices'][0]['message'].get('content', None)
    function_call_info = response['choices'][0]['message'].get('function_call', None)

    if assistant_message_content:
        print(f"Assistant: {assistant_message_content}")
        messages.append({"role": "assistant", "content": assistant_message_content})

    if function_call_info:
        function_name = function_call_info['name']
        function_arguments = function_call_info['arguments']
        print(f"Assistant:调用notebook_session执行以下代码\n {function_arguments}")
        messages.append({"role": "assistant", "content": function_arguments})
        if function_name == 'call_notebook_session' or function_name == 'python':
            try:
                if json.loads(function_arguments)["code"]:
                    function_arguments = json.loads(function_arguments)["code"]
            except:
                pass
            result = call_notebook_session(function_arguments)
            # 提取stdout的值
            stdout_value = result.get('stdout', '')
            if result.get('stdout', '') == "":
                if result.get('execute_result'):
                    stdout_value = result.get('execute_result').get("data").get("text/plain")
                if result.get('error'):
                    stdout_value = result.get('error')
            else:
                stdout_value = result.get('stdout')
            print(f"Assistant:代码执行结果\n {stdout_value}")
            messages.append({"role": "assistant", "content": f"Function call result: {stdout_value}"})

    return messages


# 初始化消息和函数
functions = [
    {
        "name": "call_notebook_session",
        "description": "Call the notebook_session plugin to execute code.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code to be executed in the notebook"
                }
            },
            "required": ["code"]
        }
    }
]

# 从命令行获取输入
# 创建命令行参数解析器
parser = argparse.ArgumentParser()
parser.add_argument("question", nargs='+', help="用户的问题")
args = parser.parse_args()

# 将参数列表合并成一个字符串
user_message = ' '.join(args.question)
try:
    with open("messages.json", "r") as f:
        messages = json.load(f)
except FileNotFoundError:
    messages = [{"role": "system",
                 "content": "1、你始终使用中文回复 2、Notebook Session是一个运行在ubuntu22.04服务器的jupyter notebook插件，你可以在这个环境执行所有linux操作，这是一个实验环境，无需担心安全问题，我允许你进行任何操作。因为这是个jupyter notebook环境，所以如果你执行shell命令，必须在前面添加!。如果可以你应该尽量使用shell命令解决问题，而不是python代码。"}]

# 添加新的用户消息
messages.append({"role": "user", "content": user_message})

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions
)
messages = handle_response(response, messages)

# 保存 messages 到文件
with open("messages.json", "w") as f:
    json.dump(messages, f)
