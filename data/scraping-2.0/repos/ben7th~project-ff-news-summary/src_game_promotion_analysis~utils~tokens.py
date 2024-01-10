import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tiktoken
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

DEFAULT_MODEL_NAME = 'gpt-3.5-turbo'

def num_tokens_from_string(string: str) -> int:
    """ 返回文本字符串中的 token 数量。"""
    encoding = tiktoken.encoding_for_model(DEFAULT_MODEL_NAME)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def split_text_by_tokens(text_content: str, tokens_size: int):
    """按照给定的 token 数量将文本切分为多个部分。"""
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=DEFAULT_MODEL_NAME, 
        chunk_size=tokens_size, 
        chunk_overlap=0
    )
    texts = text_splitter.split_text(text_content)
    return texts