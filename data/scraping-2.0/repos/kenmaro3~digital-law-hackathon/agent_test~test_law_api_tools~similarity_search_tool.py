from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import os

class QueryArgs(BaseModel):
    query: str = Field(..., description="Search query to be executed.")
    # path: str = Field(..., description="Path to the text document.")

class SimilaritySearchTool(BaseTool):
    name = "SimilaritySearchTool"
    description = "This tool loads a document, splits it into chunks, embeds each chunk and loads it into the vector store, and finally executes a similarity search."
    args_schema: Type[BaseModel] = QueryArgs  # Updated

    def _load_documents(self, path):
        return TextLoader(path).load()

    def _split_documents(self, raw_documents):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        return text_splitter.split_documents(raw_documents)

    def _create_vectorstore(self, documents):
        return Chroma.from_documents(documents, OpenAIEmbeddings())

    def _run(self, query):
        path = "tmp.txt"
        raw_documents = self._load_documents(path)
        documents = self._split_documents(raw_documents)
        db = self._create_vectorstore(documents)
        return db.similarity_search(query)
        
    async def _arun(self, query):  # Async version of _run
        return self._run(query)
