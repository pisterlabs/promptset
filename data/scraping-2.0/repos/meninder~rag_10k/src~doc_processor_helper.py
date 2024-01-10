
import os
import pandas as pd
from typing import List
from pathlib import Path, PosixPath
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from src.logger import logger

def get_all_data_paths()->List:
    pwd = Path(os.getcwd())
    path_data = pwd/'data'
    path_pdfs = [x for x in path_data.glob('**/*') if '10k' in x.name.lower()]
    logger.info(f'found {len(path_pdfs)} pdfs in {path_data}')

    return path_pdfs

def get_sub_documents(path_pdf:PosixPath, chunk_size:int=3000, chunk_overlap:int=0)->List[Document]:
    loader = UnstructuredPDFLoader(path_pdf)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(data)
    logger.info(f'{len(data)} doc inputted with {len(data[0].page_content)} chars.')
    logger.info(f'{len(texts)} sub-docs created.')
    
    return texts