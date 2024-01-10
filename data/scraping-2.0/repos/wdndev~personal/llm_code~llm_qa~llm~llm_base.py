# -*- coding: utf-8 -*-
#  @file        - llm_base.py
#  @author      - dongnian.wang(dongnian.wang@outlook.com)
#  @brief       - 在 LangChain LLM 基础上封装的项目类，统一了 GPT、文心、讯飞、智谱多种 API 调用
#  @version     - 0.0
#  @date        - 2023.12.20
#  @copyright   - Copyright (c) 2023 

from langchain.llms.base import LLM 
from typing import Dict, Any, Mapping
from pydantic import Field

class LLMBase(LLM):
    """ 自定义LLM
    """
    # 原生接口地址
    url : str = None
    # 默认选用 GPT-3.5 模型
    mode_name : str = "gpt-3.5-turbo"
    # 访问时延上限
    request_timeout : float = None
    # 温度系数
    temperature : float = 0.1
    # API key
    api_key : str = None
    # 其他参数
    mode_kwards : Dict[str, Any] = Field(default_factory=dict)

    @property
    def _default_params(self) -> Dict[str, Any]:
        """ 获取调用默认参数
        """
        normal_params = {
            "temperature" : self.temperature,
            "request_timeout" : self.request_timeout,
        }
        # print(type(self.model_kwargs))
        return {**normal_params}
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """ 获取identifying参数
        """
        return {**{"model_name": self.mode_name}, **self._default_params}
