import logging
import os
from typing import List, Optional, Union, Literal, Set, Sequence, Any

import numpy as np
from langchain.embeddings.base import Embeddings
from pydantic import BaseModel
import zhipuai

logger = logging.getLogger(__name__)


class ZhipuAiEmbeddings(BaseModel, Embeddings):
    """
    zhipuai 向量化
    """
    api_key: Optional[str] = None

    def embed_documents(
            self, texts: List[str], chunk_size: Optional[int] = 0
    ) -> List[List[float]]:
        embeddings = []
        for text in texts:

            for retry in range(3):
                # 最多重试3次
                response = zhipuai.model_api.invoke(
                    model="text_embedding",
                    prompt=text
                )
                if response is not None and 'code' in response and response["code"] == 200:
                    data = response["data"]
                    embeddings.append(data.get('embedding'))
                    break  # 成功则跳出重试循环
                else:
                    print(
                        f"Retrying: zhipu embeddings Request failed with response {response}. Retrying..., text= {text}")
                    if retry == 2:
                        print("attention!error,call zhipu embedding fail 3 times,text=", text, ';;; response=',
                              response)
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
