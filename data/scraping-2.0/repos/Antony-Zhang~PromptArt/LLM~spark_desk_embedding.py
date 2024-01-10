# -*- coding: utf-8 -*-
# @Time : 2023/7/3 15:41
# @Author : Fishead_East
# @Email : ytzd2696@foxmail.com
# @File : spark_desk_embedding.py
# @Project : PromptArt
import os
import json
import requests
from typing import Optional, List, Mapping, Any

from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.embeddings.base import Embeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from LLM.webInteract.web_param import WsParamEmb


class SparkDeskEmbedding(object):
    """
    讯飞星火的Embedding模型
    """
    url = r'https://knowledge-retrieval.cn-huabei-1.xf-yun.com/v1/aiui/embedding/query'
    APPID: str = os.getenv("APPID")
    APIKey: str = os.getenv("APIKEY")
    APISecret: str = os.getenv("APISECRET")

    def _get_param(self, text) -> Mapping[str, Any]:
        """
        组织请求消息
        :param text: 待向量化的文本
        :return:
        """
        param_dict = {
            'header': {
                'app_id': self.APPID
            },
            'payload': {
                'text': text
            }
        }
        return param_dict

    def embed_query(self, text: str,) -> List[float]:
        """Compute query embeddings using the Spark Desk Model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        ws_param = WsParamEmb(self.url, self.APPID, self.APIKey, self.APISecret)
        wsUrl = ws_param.create_url()
        param_dict = self._get_param(text)
        response = requests.post(url=wsUrl, json=param_dict)    # 得到响应串
        result_dict = json.loads(response.content.decode('utf-8'))
        embed = json.loads(result_dict['payload']['text']['vector'])
        return embed


if __name__ == '__main__':
    llm_embed = SparkDeskEmbedding()
    embed1 = llm_embed.embed_query("你好吗？")
    print(embed1)
    print(len(embed1))
    print(embed1[:5])


# class SparkDeskEmbeddings(HuggingFaceEmbeddings):
#     """重写HuggingFaceEmbeddings加载类"""
#
#     client: Any  #: :meta private:
#     model_name: str = "SparkDeskEmbeddings"
#
#     def __init__(self, **kwargs: Any):
#         super().__init__(**kwargs)
#         # self.client即向量化工具，为sentence_transformers包中的类
#
#     def embed_query(self, text: str) -> List[float]:
#         """Compute query embeddings using a HuggingFace transformer model.
#
#         Args:
#             text: The text to embed.
#
#         Returns:
#             Embeddings for the text.
#         """
#         text = text.replace("\n", " ")
#         embedding = self.client.encode(text, normalize_embeddings=True)
#         return embedding.tolist()
