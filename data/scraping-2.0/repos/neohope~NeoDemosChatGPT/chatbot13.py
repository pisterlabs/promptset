#!/usr/bin/env python3
# -*- coding utf-8 -*-

import os, yaml
import openai
import gradio as gr
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI

'''
使用ChatGPT和Gradio构建聊天机器人
增加语音输入的聊天方式
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]

# 通过chatgpt获取聊天反馈
def predict(input, history=[]):
    history.append(input)
    response = conversation.predict(input=input)
    history.append(response)
    responses = [(u,b) for u,b in zip(history[::2], history[1::2])]
    return responses, history

# 语音转文本
def transcribe(audio):
    # openai根据文件后缀名判断是否支持该格式，需要补充wav后缀名
    os.rename(audio, audio + '.wav')
    audio_file = open(audio + '.wav', "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript['text']    

# 语音先转文本，然后调用predict方法
def process_audio(audio, history=[]):
    if audio is not None:
        text = transcribe(audio)
        return predict(text, history)
    else:
        text = None
        return predict(text, history)


if __name__ == '__main__':
    get_api_key()

    # 通过SummaryBufferMemory保留之前聊天的语境
    memory = ConversationSummaryBufferMemory(llm=ChatOpenAI(), max_token_limit=2048)
    conversation = ConversationChain(
        llm=OpenAI(max_tokens=2048, temperature=0.5), 
        memory=memory,
    )

    # 页面设计，增加了语音录入方式
    with gr.Blocks(css="#chatbot{height:350px} .overflow-y-auto{height:500px}") as demo:
        chatbot = gr.Chatbot(elem_id="chatbot")
        state = gr.State([])

        # 文本录入
        with gr.Row():
            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)

        # 语音录入
        with gr.Row():
            audio = gr.Audio(source="microphone", type="filepath")
 
        # 文本关联predict方法
        txt.submit(predict, [txt, state], [chatbot, state])

        # 语音关联process_audio方法
        audio.change(process_audio, [audio, state], [chatbot, state])

    # 启动
    demo.launch()
