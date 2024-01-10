from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

class TextProcessor:
    def __init__(self, chunk_size=1200, chunk_overlap=0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print("TextProcessor Called...")

    def chunk_documents(self, documents):
        """
        Split the content of the NewsDocument objects into chunks.

        Parameters:
        - documents (list): List of NewsDocument objects.

        Returns:
        - list: List of chunked NewsDocument objects.
        """
        text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return text_splitter.split_documents(documents)

    def vectorize_chunks(self, chunks):
        """
        Vectorize the chunks using OpenAI embeddings.

        Parameters:
        - chunks (list): List of chunked NewsDocument objects.

        Returns:
        - Chroma: Vector store containing the vectorized representations of the chunks.
        """
        return Chroma.from_documents(chunks, OpenAIEmbeddings())
