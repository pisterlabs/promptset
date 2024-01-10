# 代码参考metaGPT
import os
import pickle
import urllib
from pathlib import Path
from typing import Optional
import faiss
from langchain.embeddings import logger

from langchain.vectorstores import FAISS

from document_store.base_store import LocalStore
from document_store.document import Document
from utils.embedding import get_embedding


class FaissStore(LocalStore):
    def __init__(self, raw_data: Path, cache_dir=None, meta_col='source', content_col='output'):
        self.meta_col = meta_col
        self.content_col = content_col
        super().__init__(raw_data, cache_dir)

    def _load(self) -> Optional["FaissStore"]:
        index_file, store_file = self._get_index_and_store_fname()
        if not (index_file.exists() and store_file.exists()):
            logger.info("Missing at least one of index_file/store_file, load failed and return None")
            return None
        index = faiss.read_index(str(index_file))
        with open(str(store_file), "rb") as f:
            store = pickle.load(f)
        store.index = index
        return store

    def _write(self, docs, metadatas):
        store = FAISS.from_texts(docs, get_embedding(), metadatas=metadatas)
        return store

    def get_retriever(self,**kwargs):
        return self.store.as_retriever(**kwargs)

    def persist(self):
        index_file, store_file = self._get_index_and_store_fname()
        index_file = os.path.join(self.cache_dir, index_file)
        store_file = os.path.join(self.cache_dir, store_file)
        store = self.store
        index = self.store.index
        faiss.write_index(store.index, str(index_file))
        store.index = None
        with open(store_file, "wb") as f:
            pickle.dump(store, f)
        store.index = index

    def search(self, query, expand_cols=False, sep='\n', *args, k=3, **kwargs):
        rsp = self.store.similarity_search(query, k=k)
        logger.debug(rsp)
        if expand_cols:
            content = str(sep.join([f"{x.page_content}: {x.metadata}" for x in rsp]))
            return content if len(content) < 1000 else content[:1000] + '...'
        else:
            content = str(sep.join([f"{x.page_content}" for x in rsp]))
            return content if len(content) < 1000 else content[:1000] + '...'

    def write(self):
        """根据用户给定的Document（JSON / XLSX等）文件，进行index与库的初始化"""
        if not self.raw_data.exists():
            raise FileNotFoundError
        doc = Document(self.raw_data, self.content_col, self.meta_col)
        docs, metadatas = doc.get_docs_and_metadatas()

        self.store = self._write(docs, metadatas)
        self.persist()
        return self.store

    def get_relevant_documents(self, query, **kwargs):
        return self.store.similarity_search(query, k=3, **kwargs)

    def add(self, texts: list[str], *args, **kwargs) -> list[str]:
        """FIXME: 目前add之后没有更新store"""
        return self.store.add_texts(texts)

    def delete(self, *args, **kwargs):
        """目前langchain没有提供del接口"""
        raise NotImplementedError


if __name__ == '__main__':
    faiss_store = FaissStore(Path('document_path'))
    query = input('请输入问题：')
    docs = faiss_store.search(query)
    print(docs, len(docs))
    faiss_store.add(['高等教育是什么？', '历史文化对高等教育是什么？'])
    print(faiss_store.search('高等教育是什么？'))
