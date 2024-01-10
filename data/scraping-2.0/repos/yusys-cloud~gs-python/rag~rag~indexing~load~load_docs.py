"""
@Time    : 2023/12/31 15:29
@Author  : yangzq80@gmail.com
@File    : load_docs.py
"""

import time
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document


def load_pdf(file_path) -> List[Document]:
    """Load PDF using pypdf into list of documents.

    Loader chunks by page and stores page numbers in metadata.
    """
    start = time.time()

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    print(docs,time.time()-start)

    return docs