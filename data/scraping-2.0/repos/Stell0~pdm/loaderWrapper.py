import os
from typing import List,Optional
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredEPubLoader
import time

class LoaderWrapper():
    '''Load different type of contents given in data as string. Also allow to specify metadata'''
    def __init__(self, path: str, type: str, metadata: Optional[dict] = {}):
        """Initialize with file path."""
        self.path = path
        self.type = type
        self.metadata = metadata

    def load(self) -> List[Document]:
        self.metadata["type"] = self.type
        self.metadata["timestamp"] = time.time()
        if self.type == "file":
            self.metadata["source"] = os.path.basename(self.path)
            self.metadata["filetype"] = self.path.rsplit('.', 1)[-1].lower()
            if self.metadata["filetype"] == 'txt':
                loader = TextLoader(self.path)
            elif self.metadata["filetype"] == 'pdf':
                loader = PyPDFLoader(self.path)
            elif self.metadata["filetype"] == 'epub':
                loader = UnstructuredEPubLoader(self.path)
            else:
                raise Exception("Unsupported filetype")
     
            documents = loader.load()

        # update documents metadata
        for document in documents:
            for key, value in self.metadata.items():
                document.metadata[key] = value

        return documents
