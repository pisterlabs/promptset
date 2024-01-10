import re, os
import docx2txt
from io import BytesIO

from PyPDF2 import PdfReader
import pdfplumber
from pdfminer.high_level import extract_text

# _PDF_READER = pdfplumber.open
# _PDF_READER = PdfReader
_PDF_READER = "pdfminer"

from typing import List, Union, Tuple

# LANGCHAIN
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


def parse_document(f_bytes, f_name):
    """ Parses a document into a list of Documents

    Args:
        file_object (BytesIO): The uploaded file

    Returns:
        str: The parsed document

    Raises:
        ValueError: If the file type is not supported
    """

    if f_name.endswith(".pdf"):
        return parse_pdf(f_bytes)
    elif f_name.endswith(".docx"):
        return parse_docx(f_bytes)
    elif f_name.endswith(".txt"):
        return parse_txt(f_bytes)
    else:
        raise ValueError(" ... File type not supported ... " )


def parse_docx(f_bytes: BytesIO) -> str:
    """ Parses a docx file and returns the contents as a string.

    Args:
        file (BytesIO): A file-like object containing a docx file.

    Returns:
        str: The contents of the docx file.
    """
    text = docx2txt.process(f_bytes)

    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)

    return text


def pdf_repair(text: str) -> str:
    """ Fixes common issues with pdf text extraction.

    The following operations are conducted:
        1. Merge hyphenated words
        2. Fix newlines in the middle of sentences
        3. Remove multiple newlines

    Args:
        text (str): The text extracted from a pdf file.

    Returns:
        str: The repaired text.
    """
    # Merge hyphenated words
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Fix newlines in the middle of sentences
    text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())

    ## Remove multiple newlines
    #text = re.sub(r"\n\s*\n", "\n\n", text)


    return text


def parse_pdf(f_bytes: BytesIO) -> List[str]:
    """ Parses a pdf file and returns the contents as a string.

    Args:
        file (BytesIO): A file-like object containing a txt file.

    Returns:
        str: The contents of the txt file.
    """
    # Extract text from pdf

    if "pdfminer" in _PDF_READER:
        output = extract_text(f_bytes)
    else:
        pdf = pdf_repair(_PDF_READER(f_bytes))

        # Fix common issues with pdf text extraction
        output = [pdf_repair(page.extract_text()) for page in pdf.pages]

    return output


def parse_txt(f_bytes: BytesIO) -> str:
    """ Parses a txt file and returns the contents as a string.

    Args:
        file (BytesIO): A file-like object containing a txt file.

    Returns:
        str: The contents of the txt file.
    """
    text = f_bytes.read().decode("utf-8")

    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


def text_to_docs(
        text: Union[str, List[str]],
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
        separators: Tuple[str] = ("\n\n", "\n", " ", ""),
        min_chunk_fraction=0.25,
) -> List[Document]:
    """Converts a string or list of strings to a list of Documents with metadata.

    Args:
        text (Union[str, List[str]]): A string or list of strings.
        chunk_size (int, optional): The size of each chunk.
        chunk_overlap (int, optional): The number of characters to overlap between chunks.
        separators (Tuple[str], optional): A tuple of strings to split on.

    Returns:
        List[Document]: A list of Documents.
    """
    min_chunk_size = int(min_chunk_fraction * chunk_size)

    # Take a single string as one page (coercion)
    if isinstance(text, str): text = [text]

    # Convert pages to Documents
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # - Define the text splitter to use for chunking -
    #     - RecursiveCharacterTextSplitter is a splitter that splits text into chunks of a specified size, but tries to
    #       split on a list of separators first.
    #     - This is useful for splitting text into contiguous sentences or paragraphs that have consistent semantics
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        separators=list(separators),
        chunk_overlap=chunk_overlap,
    )

    # Split each page into chunks and add page numbers and chunk numbers as metadata.
    doc_chunks = []
    for doc in page_docs:
        chunks = text_splitter.split_text(doc.page_content)
        # Add chunk numbers and `sources` as metadata
        for i, chunk in enumerate(chunks):
            if len(chunk)<min_chunk_size:
                continue
            doc = Document(page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i})
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks


def embed_text(doc_txt: str, openai_api_key: str = None) -> VectorStore:
    """Embeds a list of Documents and returns a FAISS vectorstore

    FAISS is a library for efficient similarity search and clustering of dense vectors.

    Args:
        doc_txt (str): The full document text to embed. This is kept as a string to allow for hashing
        openai_api_key (str): The OpenAI API key to use for embedding the text.

    Raises:
        AuthenticationError: If the user has not previously entered a valid OpenAI API key that is stored in state.
                             The user can enter this information in the Streamlit sidebar.
    Returns:
        VectorStore: A FAISS index of the embedded Documents.
    """

    # Convert the text to Langchain Documents
    docs = text_to_docs(doc_txt)

    # Embed the chunks
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Create vectorstore
    vs = FAISS.from_documents(docs, embeddings)
    return vs


def search_docs(vectorstore: VectorStore, query: str, top_k: int = 5) -> List[Document]:
    """Searches a FAISS index for similar chunks to the query and returns a list of Documents.

    The `query` is embedded and then compared to the embedded Documents in the vectorstore`index`.
    The `top_k` most similar Documents are then returned.

    FAISS is a library for efficient similarity search and clustering of dense vectors.

    Args:
        vectorstore (VectorStore): A FAISS index (vectorstore) of the embedded Documents.
        query (str): A query string.
        top_k (int): The number of similar chunks to return.

    Returns:
        List[Document]: A list of Documents that are similar to the query based on the embedded vector similarity.
    """

    # Search for similar chunks
    docs = vectorstore.similarity_search(query, k=top_k)
    return docs


def process_document(path):
    """ Parses the document located at the given path and returns Langchain documents.

    Args:
        path (str): The path to the document/txt/pdf file.

    Returns:
        A list of Langchain documents parsed from the original document (text file or pdf).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file type is not supported.
    """

    # Check if the file exists
    if not os.path.isfile(path):
        raise FileNotFoundError(f"\n... File not found at {path} ...\n")

    # Read the file content as bytes
    with open(path, 'rb') as f:
        f_bytes = BytesIO(f.read())
        f_name = os.path.basename(path)

        # Parse the document into text
        doc_txt = parse_document(f_bytes, f_name)

        # If the document is parsed from a PDF, it may return a list of strings, join them
        if isinstance(doc_txt, list):
            doc_txt = '\n'.join(doc_txt)

        # Convert the text into Langchain documents
        vs = embed_text(doc_txt)

        return vs
