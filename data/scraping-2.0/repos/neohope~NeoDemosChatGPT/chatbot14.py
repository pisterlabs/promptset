#!/usr/bin/env python3
# -*- coding utf-8 -*-

import os, yaml
import openai
import gradio as gr
import azure.cognitiveservices.speech as speechsdk
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI

'''
使用ChatGPT和Gradio构建聊天机器人
增加语音输入、azure语音输出的聊天方式
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]

def get_azure_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        region = yaml_data["azure"]["region"]
        subscription = yaml_data["azure"]["subscription"]
        return region, subscription

# 播放语音
def play_voice(text):
    speech_synthesizer.speak_text_async(text)

# 通过chatgpt获取聊天反馈，并播放语音
def predict(input, history=[]):
    history.append(input)
    response = conversation.predict(input=input)
    history.append(response)
    # 语音播放
    play_voice(response)
    responses = [(u,b) for u,b in zip(history[::2], history[1::2])]
    return responses, history

# 语音转文本
def transcribe(audio):
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

    # 语音配置
    # KEY及区域
    region, subscription = get_azure_key()
    speech_config = speechsdk.SpeechConfig(subscription=subscription, region=region)
    # 选用语音
    speech_config.speech_synthesis_language='zh-CN'
    speech_config.speech_synthesis_voice_name='zh-CN-XiaohanNeural'
    # 选用扬声器
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    # 创建speech_synthesizer
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    # 页面设计
    with gr.Blocks(css="#chatbot{height:800px} .overflow-y-auto{height:800px}") as demo:
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
