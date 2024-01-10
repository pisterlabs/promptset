import json
import os
from typing import List

import dotenv
from langchain.document_loaders import TextLoader
from langchain.document_loaders.base import BaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma

from chromadb import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import chromadb
from chromadb.utils import embedding_functions

dotenv.load_dotenv()

chroma_client = chromadb.PersistentClient(
    "chromadb_psylio_kb", Settings(allow_reset=True)
)
chroma_client.reset()

embedding_function = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

chroma = Chroma(
    client=chroma_client, collection_name="psylio", embedding_function=embedding_function
)


class ScrapedLoader(BaseLoader):
    def __init__(self, file_path):
        self._file_path = file_path

    def load(self) -> List[Document]:
        docs = []

        with open(self._file_path, "r") as f:
            json_data = json.load(f)

            # remove elements where content is None
            json_data = [item for item in json_data if item["content"] is not None]

            for index, item in enumerate(json_data):
                print(
                    f"Adding document {item['url']} with content size {len(item['content'])}"
                )

                document = Document(
                    page_content=item["content"],
                    metadata={
                        "title": item["title"],
                        "url": item["url"],
                        "language": item["language"],
                    },
                )
                docs.append(document)

        return docs


def run():

    for file in ["psylio.json"]:
        print(f"Loading content from {file}")

        scraped_loader = ScrapedLoader(file)
        documents = scraped_loader.load()

        print(f"Documents loaded: {len(documents)}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
        )
        split_docs = text_splitter.split_documents(documents)

        print(f"Splitd documents: {len(split_docs)}")

        chroma.add_documents(split_docs)


if __name__ == "__main__":
    run()
