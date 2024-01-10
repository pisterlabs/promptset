from typing import Any
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.embeddings.gpt4all import GPT4AllEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever


def filter_docs(split_docs: list[Document], question: str, out_files_num=4) -> Any:
    vectorstore = Chroma.from_documents(documents=split_docs, embedding=SentenceTransformerEmbeddings())
    docs = vectorstore.similarity_search(question, k=out_files_num)

    return docs


def filter_docs2(split_docs: list[Document], question: str, out_files_num=4) -> Any:
    vectorstore = Chroma.from_documents(documents=split_docs, embedding=GPT4AllEmbeddings())
    docs = vectorstore.similarity_search(question, k=out_files_num)

    return docs


# def retrieve_docs(split_docs: list[Document], question: str) -> Any:
#     vectorstore = Chroma.from_documents(documents=split_docs, embedding=SentenceTransformerEmbeddings())
#     retriever = vectorstore.as_retriever(search_kwargs={'k': 1})
#     docs = retriever.invoke(question)
#
#     return docs
