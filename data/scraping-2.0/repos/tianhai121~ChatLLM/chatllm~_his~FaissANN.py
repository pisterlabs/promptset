#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : AI.  @by PyCharm
# @File         : FaissANN
# @Time         : 2023/4/20 17:22
# @Author       : betterme
# @WeChat       : meutils
# @Software     : PyCharm
# @Description  :

from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# ME
from meutils.pipe import *


class FaissANN(object):
    def __init__(self, folder_path: str = None,
                 index_name: str = "index",
                 model_name_or_path="shibing624/text2vec-base-chinese"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name_or_path)
        self.folder_path = folder_path
        self.faiss_ann = self.load_local(folder_path, index_name)  # 加载已存在的索引

    def add_texts(self, texts, metadatas: Optional[List[dict]] = None):  # todo: 增加进度条
        self.faiss_ann = FAISS.from_texts(texts, self.embeddings, metadatas)  # metadatas = [{'source': 'xx'}]

    def update(self, target: FAISS, index_name=None):
        self.faiss_ann.merge_from(target)  # Add the target FAISS to the current one.
        if index_name:
            logger.info('保存新的索引文件')
            self.faiss_ann.save_local(self.folder_path, index_name)

    def load_local(self, folder_path=None, index_name: str = "index"):
        if folder_path and Path(folder_path).is_dir():
            return FAISS.load_local(folder_path, self.embeddings, index_name)
