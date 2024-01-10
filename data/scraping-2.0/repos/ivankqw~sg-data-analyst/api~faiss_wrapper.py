from typing import List
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
import time
import termcolor
from pprint import pprint
from definitions.dataset import Dataset

color = "magenta"


def split_and_store(dataset_docs: List[Document]) -> FAISS:
    """Splits the documents into chunks and stores them into the vector store.
    
    :param dataset_docs: the dataset documents :class `List[Document]`

    :return: the vector store :class `FAISS`
    """
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    print(f"Splitting and embedding {len(dataset_docs)} documents.")
    dataset_split_docs = text_splitter.split_documents(dataset_docs)
    print(f"Loading split documents into vector store.")
    start = time.time()
    vector_store = FAISS.from_documents(dataset_docs, OpenAIEmbeddings())
    # TODO: Bottleneck
    print(termcolor.colored(
        f"Loaded documents into vector store in {time.time() - start} seconds.", color))
    return vector_store

def similarity_search(db: FAISS, query: str) -> List[Document]:
    """Performs a similarity search on the vector store.

    :param db: the vector store :class `FAISS`
    :param query: the query :class `str`

    :return: the fetched documents :class `List[Document]`
    """
    fetched_docs = db.similarity_search(query)
    print(termcolor.colored(f"Found {len(fetched_docs)} candidate documents.", color))
    pprint(termcolor.colored(f"Top 5 documents that match the query {query}: {[Dataset.from_document(x).name for x in fetched_docs[:5]]}", color))
    return fetched_docs