from typing import Any
from langchain.document_loaders import TextLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


def filter_docs(split_docs: list[Document], question: str) -> Any:
    vectorstore = Chroma.from_documents(documents=split_docs, embedding=GPT4AllEmbeddings())
    docs = vectorstore.similarity_search(question)
    return docs
