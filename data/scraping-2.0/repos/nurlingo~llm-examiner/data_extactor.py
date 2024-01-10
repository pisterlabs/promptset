import os
import shutil
import json

from langchain.document_loaders import (
    BSHTMLLoader,
    DirectoryLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from annotations import Annotations as A

from dotenv import load_dotenv


def extract_data_from_webpage(knowledge_base_id: A.knowledge_base_id,
                              url: A.url):
    load_dotenv()
    # Define the directory path
    chroma_directory = f"./chroma/{knowledge_base_id}"

    if os.path.exists(chroma_directory):
        print("Deleting the existing Chroma directory...")
        shutil.rmtree(chroma_directory)
        print("Directory deleted.")

    loader = DirectoryLoader(
        f"./scrape/{knowledge_base_id}/",
        glob="*.html",
        loader_cls=BSHTMLLoader,
        show_progress=True,
        loader_kwargs={"get_text_separator": " "},
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    data = loader.load()
    documents = text_splitter.split_documents(data)

    # map sources from file directory to web source
    with open(f"./scrape/{knowledge_base_id}/sitemap.json", "r") as f:
        sitemap = json.loads(f.read())

    for document in documents:
        document.metadata["source"] = sitemap[
            document.metadata["source"].replace(".html", "").replace(f"scrape//{knowledge_base_id}/", "")
        ]

    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    db = Chroma.from_documents(
        documents,
        embedding_model,
        persist_directory=f"./chroma/{knowledge_base_id}",)
    db.persist()
    return True

# def call_from_terminal(knowledge_base_id, url):
#     print(f"Argument 1: {knowledge_base_id}")
#     print(f"Argument 2: {url}")
#     extract_data_from_webpage(knowledge_base_id, url)

# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: python my_script.py arg1 arg2")
#     else:
#         knowledge_base_id = sys.argv[1]
#         url = sys.argv[2]
#         call_from_terminal(knowledge_base_id, url)