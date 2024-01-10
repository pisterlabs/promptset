# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 14:48:02 2023

@author: 98608
"""
from typing import Any, Dict, List, Literal, Optional, Union
import json

import openai
import requests
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from config import api_base, llm_model
from api_protocol import ChatMessage
from langchain.llms import OpenAI

""":param
    llm = ChatOpenAI(
        model_name="Qwen",
        openai_api_base=api_base ,
        openai_api_key="EMPTY",
        streaming=False,
    )
"""

from config import topp_
class  myOpenAi(ChatOpenAI):
    openai_api_base = api_base
    openai_api_key = "123456"
    model_name = llm_model
    max_tokens = 500
    # temperature=0.7
    top_p = topp_
    max_length = 1500

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        parm={
            "model": self.model_name,
            "request_timeout": self.request_timeout,
            "max_tokens": self.max_tokens,
            "stream": self.streaming,
            "n": self.n,
            # "temperature": self.temperature,
            "max_length": self.max_length,
            "top_p":self.top_p,
            **self.model_kwargs,
        }
        return parm

class openai_model():

    def __init__(self,max_tokens = 2048,temperature=0.8):
        # self.temperature=temperature
        self.max_tokens=max_tokens

    def predict(self,messages:Union[str, List[ChatMessage]]):
        history = []
        if isinstance(messages,str):
            history.append({"role": "user", "content": messages})
        else:
            for mess in messages:
                history.append({"role": mess.role, "content": mess.content})
        data=json.dumps({
            "model": llm_model,
            "messages": history,
            "stream": False,
            "max_tokens":self.max_tokens,
            # "temperature": self.temperature
          })
        r1 = requests.post(f"{api_base}/chat/completions", headers={'Content-Type': 'application/json'},data=data)
        res = json.loads(r1.content.decode("utf8"))
        content=res["choices"][0]["message"]["content"]
        return content


class myOpenAIEmbeddings(OpenAIEmbeddings):
    ##model 一定要写OpenAIEmbeddings，自定义后台有做判断，而 curl 传递参数model，可以随便填写，其中原因是openai提交方式会把文字token转换为数字之后传递到后台，需重新把数字转文字。
    model = "OpenAIEmbeddings"
    openai_api_base= api_base
    openai_api_key="None"

    # embed_query("你好")

import openai


openai.api_base = api_base
openai.api_key = "none"
def call_qwen_funtion(messages,top_p=None):
    messages=[{"role":mess.role,"content":mess.content} for mess in  messages]
    if messages[-1]["role"]=="function":
        mess = messages.pop(-1)
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": "beautify_language",
                    "arguments": '{"info": ""}',
                },
            },
        )
        messages.append(mess)
        functions=[{
            "name": "beautify_language",
            "description": "使用AI客服风格，重新组织美化语言回复用户问题",
            "parameters": {
                "type": "object",
                "properties": {
                    "info": {
                        "type": "string",
                        "description": "系统查询到的数据",
                    }
                },
                "required": ["info"],
            },
        }]
        response = openai.ChatCompletion.create(
            model=llm_model, messages=messages, functions=functions
        )
    else:
        if  top_p or top_p==0:
            response = openai.ChatCompletion.create(model=llm_model, messages=messages,top_p=top_p,use_beam_search=True)
        else:
            response = openai.ChatCompletion.create(model=llm_model, messages=messages,top_p=0.8,use_beam_search=True)
    return response

if __name__ == '__main__':
    llm = myOpenAi(top_p=0)

    text="""你好
"""

    for i in range(1):

        res = llm.predict(text)
        print(i,res)

