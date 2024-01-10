# -*- coding: utf-8 -*-
#  @file        - chatGLM3_llm.py
#  @author      - dongnian.wang(dongnian.wang@outlook.com)
#  @brief       - 基于fastapi的ChatGLM3 自定义类
#  @version     - 0.0
#  @date        - 2023.12.20
#  @copyright   - Copyright (c) 2023 



BASE_URL = "http://172.16.14.79:6006"

from typing import Any, List, Mapping, Optional, Dict, Union, Tuple
from pydantic import Field
import json
import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun

from llm.llm_base import LLMBase

class ChatGLM3LLM(LLMBase):
    """ 基于fastapi的ChatGLM3 自定义类
    """
    def _call(self, 
              prompt : str, 
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        
        # 配置 POST 参数
        payload = json.dumps({
            "prompt": prompt,
            "top_p" :  0.8,
            'temperature' : self.temperature
        })
        headers = {
            'Content-Type': 'application/json'
        }

        # 发起请求
        response = requests.request("POST", BASE_URL, headers=headers, data=payload, timeout=self.request_timeout)
        if response.status_code == 200:
            # 返回的是一个 Json 字符串
            js = json.loads(response.text)
            # print(js)
            return js["result"]
        else:
            return "请求失败"
    
    @property
    def _llm_type(self) -> str:
        return "ChatGLM3.0"

