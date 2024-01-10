"""
Documents are loaded into memory
"""

from typing import List
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.docstore.document import Document

def load_data() -> List[Document]:
    """
    The function that loads the data.
    """
    loader = DirectoryLoader('./docs/', glob="**/*.txt", loader_cls=TextLoader,
                            loader_kwargs={'autodetect_encoding': True}, 
                            )
    try:
        docs = loader.load()
        print(f"{len(docs)} documents loaded.")
    except:
        print("Error loading documents.")
        raise
    return docs
