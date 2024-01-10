from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from fastapi import HTTPException
import random


async def load_data(path, file_type, max_pages=1000, is_keyword=False):
    pages = []
    if file_type == 'application/pdf':
        loader_first = PyPDFLoader(path)
        pages = loader_first.load_and_split()
    elif file_type == 'text/plain':
        loader_first = TextLoader(path, encoding='utf-8')
        pages = loader_first.load_and_split()
    elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        loader_first = Docx2txtLoader(path)
        pages = loader_first.load()
    else:
        raise ValueError("File type %s is not a valid file type" % file_type)
    if is_keyword and len(pages) > 2*max_pages:
        prefix_list = pages[:max_pages]
        sub_list = pages[max_pages:]
        suffix_list = random.sample(sub_list, max_pages)
        return prefix_list + suffix_list
    else:
        return pages[:max_pages]


async def lazy_load_data(path, file_type):
    if file_type == 'application/pdf':
        loader_first = PyPDFLoader(path)
        pages = loader_first.load()
        return pages
    elif file_type == 'text/plain':
        loader_first = TextLoader(path, encoding='utf-8')
        pages = loader_first.load()
        return pages
    elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        loader_first = Docx2txtLoader(path)
        pages = loader_first.load_and_split()
        return pages
    else:
        raise ValueError("File type %s is not a valid file type" % file_type)
