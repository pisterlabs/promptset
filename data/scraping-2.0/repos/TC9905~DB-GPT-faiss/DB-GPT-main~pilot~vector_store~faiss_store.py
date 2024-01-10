import os
import logging
import shutil
import numpy as np
from typing import Any


# import faiss
import faiss
from langchain.vectorstores import FAISS
from langchain.schema import Document
from pilot.vector_store.base import VectorStoreBase

logger = logging.getLogger(__name__)

class FaissStore(VectorStoreBase):
    """Faiss database"""

    #加载数据库，获取索引index
    #ids = list(self.vector_store_client.docstore.__dict__.keys())无法运行，ids列表获取失败,无法删除ids=0的默认值
    def __init__(self, ctx: {}) -> None:
        
        self.ctx = ctx
        self.embeddings = ctx.get("embeddings", None)
        self.persist_dir = os.path.join(
            ctx["chroma_persist_path"], ctx["vector_store_name"] + ".vectordb"
        )

        if os.path.isfile(os.path.join(self.persist_dir, "index.faiss")):
            embeddings = self.embeddings
            self.vector_store_client = FAISS.load_local(self.persist_dir, embeddings)
        else:
             # create an empty vector store
            if not os.path.exists(self.persist_dir):
                os.makedirs(self.persist_dir)
            doc = Document(page_content="init", metadata={})
            self.vector_store_client = FAISS.from_documents([doc], self.embeddings)
            # ids = list(self.vector_store_client.docstore.__dict__.keys())
            # print("提取ids 完成")
            # self.vector_store_client.delete(ids)
            self.vector_store_client.save_local(self.persist_dir)
            self.vector_store_client = FAISS.load_local(self.persist_dir, self.embeddings)
            # self.vector_store_client.save_local(self.persist_dir)

    def similar_search(self, text, topk, **kwargs: Any) -> None:
        logger.info("FaissStore similar search")
        return self.vector_store_client.similarity_search(text, topk)

    def vector_name_exists(self):
        logger.info(f"Check persist_dir: {self.persist_dir}")
        return (
            os.path.exists(self.persist_dir) and len(os.listdir(self.persist_dir)) > 0
        )

        # if not os.path.exists(self.persist_dir):
        #     return False
        # files = os.listdir(self.persist_dir)
        # files = list(filter(lambda f: f != "faiss.sqlite3", files))

        # return len(files) > 0

        # logger.info(f"Check persist_dir: {self.persist_dir}")
        # if not os.path.exists(self.persist_dir):
        #     return False
        # files = os.listdir(self.persist_dir)
        # files = list(filter(lambda f: f != "faiss.sqlite3", files))
        # return len(files) > 0

    def load_document(self, documents):
        logger.info("FaissStore load document")
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = self.vector_store_client.add_texts(texts=texts, metadatas=metadatas)
        self.vector_store_client.save_local(self.persist_dir)
        return ids

    def delete_vector_name(self, vector_name):
        logger.info(f"faiss vector_name:{vector_name} begin delete...")
        if self.vector_name_exists():
            shutil.rmtree(self.persist_dir)
            logger.info(f"faiss vector_name:{vector_name} deleted.")
            return True
        else:
            logger.warning(f"Vector name '{vector_name}' not found.")
            return False

    def delete_by_ids(self, ids):
        print("begin delete faiss ids...")
        logger.info(f"begin delete faiss ids...")
        self.vector_store_client.delete(self, ids)
        print("delete faiss ids finished")
