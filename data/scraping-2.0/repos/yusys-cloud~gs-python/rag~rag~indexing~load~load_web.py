"""
@Time    : 2023/12/30 23:04
@Author  : yangzq80@gmail.com
@File    : load_web.py
"""

from langchain.document_loaders import TextLoader,WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = WebBaseLoader('https://python.langchain.com/docs/modules/data_connection/')

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

docs=loader.load_and_split(text_splitter)

print(len(docs))

