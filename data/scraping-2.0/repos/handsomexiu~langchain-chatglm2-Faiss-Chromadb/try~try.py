import re
from typing import List
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
loader = UnstructuredFileLoader(
    "try/1.pdf", strategy="fast", mode="elements"
)
docs = loader.load()
# 我执行这个的时候自动帮我安装了ntlk_data，
 