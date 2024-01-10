import os
import json

from langchain.document_loaders import (
    BSHTMLLoader,
    DirectoryLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from dotenv import load_dotenv

root_path = os.path.sep.join(os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-2])
import os, sys
sys.path.append(root_path)
from configurations.config_llm import OPENAI_API_KEY

def embed(files_path, db_path, web = False):
    load_dotenv()

    if web:
        loader = DirectoryLoader(
            files_path,
            glob="*.html",
            loader_cls=BSHTMLLoader,
            show_progress=True,
            loader_kwargs={"get_text_separator": " ", "open_encoding": "latin-1"},
        )
    else:
        loader = DirectoryLoader(
            files_path,
            glob="*.pdf",
            show_progress=True,
            loader_kwargs={"get_text_separator": " ", "open_encoding": "latin-1"},
        )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    data = loader.load()
    documents = text_splitter.split_documents(data)

    if web:
        # map sources from file directory to web source
        with open(f"{files_path}/sitemap.json", "r") as f:
            sitemap = json.loads(f.read())

        for document in documents:
            document.metadata["source"] = sitemap[document.metadata["source"].replace(".html", "").replace(f"{files_path}/", "")]
    else:
        for document in documents:
            document.metadata["source"] = document.metadata["source"].replace(".pdf", "").replace(f"{files_path}/", "")

    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)

    if os.path.exists(db_path):
        db = Chroma(persist_directory=db_path, embedding_function=embedding_model)
        db.add_documents(documents)
    else:
        db = Chroma.from_documents(documents, embedding_model, persist_directory=db_path)
        db.persist()