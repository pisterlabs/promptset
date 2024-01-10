#encoding=utf8

import os
import json
import requests
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

class Text2MusicTool(object):
    _name_ = "Text2Music"
    _description_ = "这个工具是从文本生成音乐的调用接口，它可以根据一段文字，生成符合这段文字内容的音乐风格。本工具的输入是一段文本指令。This tool is an API that generates music from text. It can create music that matches the style of the given text content. The input for this tool is a text command."

    def __init__(self):
        self.translator = ChatOpenAI(
            model="gpt-3.5-turbo",
            max_tokens=256)
        self.host = "0.0.0.0"
        self.port = 6678
    
    def inference(self, input_str: str):
        messages = []
        messages.append(
            SystemMessage(
                content=[
                    {"type": "text", "text": "你是一个翻译专家，请将我输入的中文翻译成英文。"}
                ]
            )
        )
        messages.append(HumanMessage(content=input_str))

        response_msg = self.translator.invoke(messages)
        input_str_en = response_msg.content
        # print(input_str_en)

        url = f"http://{self.host}:{self.port}/text_to_music"
        data = {"text": input_str_en}
        music_response = requests.post(url, data=data)
        return music_response.text