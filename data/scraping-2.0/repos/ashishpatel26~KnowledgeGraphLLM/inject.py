from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# TODO: 
# [X]1. Document loaders
# [X]2. Text Splitter
# [X]3. Embeddings
# [X]4. Vector stores

class DocLoad:
    """The `DocLoad` class is responsible for loading and splitting PDF documents."""

    def __init__(self, file_path = None):
        """
        Constructor that initializes an object with a file path.

        Args:
            file_path (str): String representing the path to a file.
        """
        # COMMENT: Initialize the object with a file path
        self.file_path = file_path
        self.loader = None
        self.document = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.chunk = None

    def load_pdf(self):
        """
        Loads a PDF file using PyPDFLoader and returns the loaded document.

        Returns:
            Loaded document.
        """
        # COMMENT: This function loads a PDF file using PyPDFLoader and returns the loaded document.
        print("[Status] Loading PDF...✅✅✅")
        self.loader = PyMuPDFLoader(self.file_path)
        self.document = self.loader.load()
        print("[Status] Loading PDF and split...✅✅✅")
        return self.document

    def split_text(self):
        """
        Loads a PDF document, splits it into chunks of text, and returns the chunks.
        """
        # COMMENT: This function is text splitting used for document
        if not self.document:
            self.load_pdf()

        self.chunk = self.text_splitter.split_documents(documents=self.document)
        return self.chunk
    
class VectorStore(FAISS):
    '''
    VectorStore using FAISS
    '''
    def __init__(self, embeddings=None, docs=None):
        """
        Initialize a VectorStore object.

        Args:
            embeddings (Optional): Embeddings for the vector store.
            docs (Optional): Documents for the vector store.
        """
        super().__init__()
        self.embeddings = embeddings
        self.docs = docs
        self.vectordb = None

    def load_faiss(self):
        """
        Load the FAISS vector store from documents and embeddings.

        Returns:
            FAISS vector database.
        """
        # COMMENT: This is FAISS vector store from documents and embeddings.
        self.vectordb = FAISS.from_documents(self.docs, self.embeddings)
        return self.vectordb

        
if __name__ == '__main__':
    doc = DocLoad(file_path="./data/tmp/tmp36hqd4ss.pdf")
    chunks = doc.split_text()
    vectordb = VectorStore(embeddings=OpenAIEmbeddings(), docs=chunks)

    
