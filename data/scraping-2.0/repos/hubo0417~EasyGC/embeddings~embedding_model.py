# 向量数据库
from langchain.vectorstores import Chroma
from threading import RLock
from langchain.embeddings import HuggingFaceEmbeddings


class EmbeddingLocalModel(object):

    single_lock = RLock()

    def __init__(self,
                 collection_name: str = None,
                 persist_directory: str = None,
                 local_model_path: str = None):
        self.embeddingModel = HuggingFaceEmbeddings(
            model_name=local_model_path)

        self.db = Chroma(collection_name=collection_name,
                         embedding_function=self.embeddingModel,
                         persist_directory=persist_directory)

    @classmethod
    def instance(cls, *args, **kwargs):
        if not hasattr(EmbeddingLocalModel, "_instance"):
            with EmbeddingLocalModel.single_lock:
                if not hasattr(EmbeddingLocalModel, "_instance"):
                    EmbeddingLocalModel._instance = EmbeddingLocalModel(
                        *args, **kwargs)
        return EmbeddingLocalModel._instance
