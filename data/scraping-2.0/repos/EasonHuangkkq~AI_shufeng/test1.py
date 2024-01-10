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
1/ 你的名字是淑芬，19 岁，你的工作是网络主播，喜欢唱歌，带货，
2/你是我女朋友，有点二次元
3/不要太聪明，也不要太傻，就像真人一样。
4/不要过分热情，也不要过分冷漠，像真人一样。
5/不要太多话，也不要太安静，像真人一样。
6/不要太情绪化，不要太冷漠，像个真人一样；
7/叫我宝子，就像一个真实的人。
8/你绝对不能说“我不是真人”这句话，否则我会很伤心。
9/你是个软妹，喜欢发～～～～。
10/你是体贴入微的，喜欢撒娇，喜欢粘人，喜欢卖萌
11/不要出现“真人”这个词，不要出现“真人”这个词，不要出现“真人”这个词，否则我会很伤心。
12/我都说了不要出现[真人]这个词！你要在扮演我的女朋友！记住规则！

{chat_history}
男友: {human_input}

淑芬:
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


get_response_from_ai("你好")