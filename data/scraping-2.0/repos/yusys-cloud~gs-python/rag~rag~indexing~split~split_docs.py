"""
@Time    : 2023/12/31 16:01
@Author  : yangzq80@gmail.com
@File    : split_docs.py
"""
from typing import List,Iterable
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

def split_docs( documents: Iterable[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    all_splits = text_splitter.split_documents(documents)
    logger.info(f'documents: {len(documents)} --> chunks: {len(all_splits)}')
    return all_splits