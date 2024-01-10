# coding=utf-8
# /usr/bin/evn python3
"""
@date: 2023-02-06
@author: HangHangLi
@version: v1.0.0
@description: 基于openai的SDK调用语言模型，如GPT-3，实现提取，摘要，问答等任务。
参考文档：
1. 0307 https://hujialou.quip.com/QynAAneV0Zcv/2023-03-07-ChatGPT- 内部业务场景
"""
import os
import sys

import openai
import gradio as gr

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.insert(0, os.path.split(rootPath)[0])
from web import openai_key, host
from data import example, prompt_text

openai.api_key = openai_key
model_type = 'MemFinLLM'


def extract(text: str, task_type) -> str:
    """
    调用模型
    Args:
        text: 输入文本
        task_type: 定义提取schema

    Returns:

    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # 选择模型
        temperature=0.6,
        max_tokens=1024,
        messages=[{"role": "user", "content": generate_prompt(text, task_type)}]
    )
    return response.choices[0]['message']['content']


def generate_prompt(text, task_type):
    """
    生成提示信息
    Returns:
    """
    return f"{prompt_text[task_type]} {text.strip() }\n"


demo = gr.Interface(
        fn=extract,
        inputs=[
            gr.Textbox(lines=10, label="Text", max_lines=20),
            gr.Radio(
                choices=list(prompt_text.keys()),
                label="任务类型",
                value='问答'
            )
        ],
        outputs='text',
        description=f'Model by {model_type}',
        examples=example,
        title="NLP应用场景演示",
        allow_flagging='never',
        css="footer {visibility: hidden}"
    )


demo.launch(server_name=host, share=True)
