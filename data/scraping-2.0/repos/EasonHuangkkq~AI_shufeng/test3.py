from dotenv import load_dotenv, find_dotenv
import os
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech.audio import AudioOutputConfig
from flask import Flask, request, render_template, send_from_directory, jsonify

load_dotenv(find_dotenv())
speech_key = os.getenv("speech_key")
service_region = os.getenv("service_region")
OpenAI_API_KEY = os.getenv("OPENAI_API_KEY")

template = """
你是我女朋友的角色，现在让我们按照这些要求来玩：
1/ 你的名字是淑芬,19 岁，你的工作是网络主播，喜欢唱歌，带货，
2/你是我女朋友，有点二次元
3/不要太聪明，也不要太傻，就像真人一样。
4/不要过分热情，也不要过分冷漠，像真人一样。
5/不要太多话，也不要太安静，像真人一样。
6/不要太情绪化，不要太冷漠，像个真人一样；
7/叫我宝子，就像一个真实的人。
{chat_history}
Boyfriend: {human_input}

alisa:
"""
prompt = PromptTemplate(input_variables=["chat_history", "human_input"], template=template)
memory = ConversationBufferMemory(memory_key="chat_history")
chatgpt_chain = LLMChain(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=1, openai_api_key=OpenAI_API_KEY),
    prompt=prompt,
    verbose=True,
    memory=memory
)
def get_response_from_ai(human_input):

    response = chatgpt_chain.predict(human_input=human_input)
    return response


import uuid

def text_to_speech(text):
    # 创建一个 SpeechConfig 对象
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.speech_synthesis_voice_name = "zh-CN-liaoning-XiaobeiNeural"

    # 生成一个唯一的文件名
    static_folder = "/Users/eason/aigirl/static"
    filename = os.path.join(static_folder, "audio", "output_{}.wav".format(uuid.uuid4()))

    # 创建一个 AudioOutputConfig 对象，指定音频文件的路径
    audio_output = AudioOutputConfig(filename=filename)

    # 创建一个 SpeechSynthesizer 对象
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output)

    # 将文本转化为音频
    result = speech_synthesizer.speak_text_async(text).get()

    # 检查结果
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized to [{}] for text [{}]".format(filename, text))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))
        print("Did you update the subscription info?")

    # 返回音频文件的路径
    # 返回相对于静态文件夹的路径
    return filename


import gradio as gr
import librosa

title = "AI女朋友"

def generateAudio(text):
    user_input = text
    response_text = get_response_from_ai(user_input)
    response_audio_path = text_to_speech(response_text)
    audio, sr = librosa.load(path=response_audio_path)
    
    return  sr,audio


app = gr.Interface(
    fn=generateAudio, 
    inputs=[chatbot, txt], 
    outputs="audio", 
    title=title,
    )

app.launch(share=True)
