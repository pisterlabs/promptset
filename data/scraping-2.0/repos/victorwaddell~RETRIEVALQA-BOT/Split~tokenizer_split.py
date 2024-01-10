from langchain.text_splitter import RecursiveCharacterTextSplitter
from Setup.tokenizer_variables import tiktoken_len
from Setup.split_variables import chunk_size, chunk_overlap

token_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap,
                                               length_function = tiktoken_len,
                                               separators = ["\n\n", "\n", " ", ""])