from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.logs import add_log

import tiktoken
tiktoken.encoding_for_model('gpt-3.5-turbo')
tokenizer = tiktoken.get_encoding('cl100k_base')


def tiktoken_len(text):
    # add_log("tokenization via tiktoken")
    tokens = tokenizer.encode(
        text, 
        disallowed_special=()
    )
    return len(tokens)


def get_text_chunks(raw_text):
    add_log("chuking via RecursiveCharacterTextSplitter")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["/n/n", "/n", " ", ""],
        chunk_size=400,
        chunk_overlap=20,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks
