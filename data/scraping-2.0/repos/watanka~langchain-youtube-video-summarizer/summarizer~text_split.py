from typing import List

import tiktoken
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


tokenizer = tiktoken.get_encoding('cl100k_base')

def token_size(text) :
    tokens = tokenizer.encode(text)
    return len(tokens)

def split_docs(context : str) -> List[Document] :
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 128,
        chunk_overlap = 25, 
        length_function = token_size
    )

    docs = [Document(page_content = x) for x in text_splitter.split_text(context)]

    return docs