import re
from itertools import chain
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredURLLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.vectorstores.base import VectorStoreRetriever


def load_text_file(file_path: str) -> Document:
    """Loads a text file and returns a Document object.

    Args:
        file_path: Path to the text file.

    Returns:
        A Document object.
    """
    doc = TextLoader(file_path, encoding="utf-8").load()[0]
    return doc


def load_pdf_file(file_path: str) -> List[Document]:
    """Loads a pdf file and returns a list of Document objects.

    Args:
        file_path: Path to the pdf file.

    Returns:
        A list of Document objects. Every page in the pdf file is a Document object.
    """
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs


def load_website(url: str) -> List[Document]:
    """Loads a website and returns a Document object.

    Args:
        url: Url of the website.

    Returns:
        A Document object.
    """
    documents = UnstructuredURLLoader(
        [url],
        mode="elements",
        headers={
            "ssl_verify": "False",
        },
    ).load()

    processed_docs = []

    # We are not rich, we need to eliminate some of the elements
    for doc in documents:
        # This will make us lose table information sorry about that :(
        if doc.metadata.get("category") not in [
            "NarrativeText",
            "UncategorizedText",
            "Title",
        ]:
            continue

        # Remove elements with empty links, they are mostly recommended articles etc.
        if doc.metadata.get("links"):
            link = doc.metadata["links"][0]["text"]
            if link is None:
                continue

            link = link.replace(" ", "").replace("\n", "")
            if len(link.split()) == 0:
                continue

        # Remove titles with links, they are mostly table of contents or navigation links
        if doc.metadata.get("category") == "Title" and doc.metadata.get("links"):
            continue

        # Remove extra spaces
        doc.page_content = re.sub(" +", " ", doc.page_content)

        # Remove docs with less than 3 words
        if len(doc.page_content.split()) < 3:
            continue

        processed_docs.append(doc)

    #  Instead of splitting element-wise, we merge all the elements and split them in chunks
    merged_docs = "\n".join([doc.page_content for doc in processed_docs])
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    processed_docs = splitter.split_text(merged_docs)
    processed_docs = [
        Document(page_content=doc, metadata={"url": url}) for doc in processed_docs
    ]

    return processed_docs


def load_text_files(file_paths: List[str]) -> List[Document]:
    """Loads a list of text files and returns a list of Document objects.

    Args:
        file_paths: List of paths to the text files.

    Returns:
        A list of Document objects.
    """
    docs = [load_text_file(file_path) for file_path in file_paths]
    return docs


def load_pdf_files(file_paths: List[str]) -> List[Document]:
    """Loads a list of pdf files and returns a list of Document objects.

    Args:
        file_paths: List of paths to the pdf files.

    Returns:
        A list of Document objects. Every page in the pdf file is a Document object.
    """
    docs = [load_pdf_file(file_path) for file_path in file_paths]
    docs = list(chain.from_iterable(docs))
    return docs


def create_index(docs: List[Document]) -> VectorStoreRetriever:
    """Creates a vectorstore index from a list of Document objects.

    Args:
        docs: List of Document objects.

    Returns:
        A vectorstore index. It searches the most similar document to the given query but with
        the help of MMR it also tries to find the most diverse document to the given query.

    """
    index = VectorstoreIndexCreator(
        vectorstore_cls=DocArrayInMemorySearch,
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        ),
    ).from_documents(docs)

    return index.vectorstore.as_retriever(search_type="mmr")
