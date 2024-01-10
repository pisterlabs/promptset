import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader

open_api_key = "<open-ai key>"

os.environ['OPENAI_API_KEY'] = open_api_key

embedding = OpenAIEmbeddings(openai_api_key=open_api_key)

OUT_FILE_DIR = "app/static"


def save_pdf_vec(pdf_location: str, uuid: str):
    """
    converts the pdf at pdf_location to a vector and saves it to the static folder.
    returns the vectorstore object.

    :param pdf_location: the location of the pdf to be converted.
    :param uuid: the uuid of the vectorstore.
    :return: the vectorstore object.

    """
    loader = PyPDFLoader(pdf_location)
    pages = loader.load_and_split()
    TEXT_SPLITTER = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )
    texts = TEXT_SPLITTER.split_documents(pages)
    docsearch = FAISS.from_texts(
        [t.page_content for t in texts],
        embedding=embedding)
    docsearch.save_local(
        f"{OUT_FILE_DIR}/{uuid}",
        uuid
    )
    os.remove(pdf_location)
    return docsearch


def load_vec(uuid: str):
    return FAISS.load_local(
        f"{OUT_FILE_DIR}/{uuid}/",
        embeddings=embedding,
        index_name=uuid
    )
