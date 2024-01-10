# -*- coding: utf-8 -*-
#  @file        - zhupuai_embedding.py
#  @author      - dongnian.wang(dongnian.wang@outlook.com)
#  @brief       - 读取文件，创建向量数据库
#  @version     - 0.0
#  @date        - 2023.12.20
#  @copyright   - Copyright (c) 2023 

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, root_validator
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class ZhipuAIEmbeddings(BaseModel, Embeddings):
    """ 智普ai Embedding
    """
    # api key
    zhipuai_api_key: Optional[str] = None

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """
        验证环境变量或配置文件中的zhipuai_api_key是否可用。

        Args:
            values: 包含配置信息的字典必须包含zhipuai_api_key字段
        Returns:
            如果环境变量或配置文件中未提供zhipuai_api_key，
            则返回原始值;否则，返回包含zhipuai_api_key的值。
        Raises:

            ValueError: zhipuai 包未找到，请安装 `pip install zhipuai`
        """

        values["zhipuai_api_key"] = get_from_dict_or_env(
            values,
            "zhipuai_api_key",
            "ZHIPUAI_API_KEY",
        )

        try:
            import zhipuai
            zhipuai.api_key = values["zhipuai_api_key"]
            values["client"] = zhipuai.model_api

        except ImportError:
            raise ValueError(
                "Zhipuai package not found, please install it with "
                "`pip install zhipuai`"
            )
        return values

    def _embed(self, texts: str) -> List[float]:
        # send request
        try:
            resp = self.client.invoke(
                model="text_embedding",
                prompt=texts
            )
        except Exception as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        if resp["code"] != 200:
            raise ValueError(
                "Error raised by inference API HTTP code: %s, %s"
                % (resp["code"], resp["msg"])
            )
        embeddings = resp["data"]["embedding"]
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embedding 文本

        Args:
            Text (str): 要嵌入的文本。

        Return:
            List [float]: 输入文本的嵌入列表，其中包含一系列浮点数值。
        """
        resp = self.embed_documents([text])
        return resp[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        嵌入文档

        Args:
            texts (List[str]): 要嵌入的文本文档列表。

        Returns:
            List[List[float]]: 输入列表中每个文档的嵌入列表。每个嵌入都表示为一个浮点数值列表。
        """
        return [self._embed(text) for text in texts]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        raise NotImplementedError(
            "Please use `embed_documents`. Official does not support asynchronous requests")

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        raise NotImplementedError(
            "Please use `aembed_query`. Official does not support asynchronous requests")
