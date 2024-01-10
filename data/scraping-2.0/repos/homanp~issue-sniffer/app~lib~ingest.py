import os

import pinecone
from decouple import config
from langchain.document_loaders import GitLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone

from app.lib.constants import FILE_PATH_WHITELIST

pinecone.init(
    api_key=config("PINECONE_API_KEY"),
    environment=config("PINECONE_ENVIRONMENT"),
)


async def ingest(body: dict) -> None:
    """Ingest a Github repo to Pinecone"""
    for repository in body["repositories"]:
        repo_id = repository["id"]
        repo_name = repository["full_name"]
        repo_url = f"https://github.com/{repo_name}"
        repo_path = f"./repos/{repo_id}/"
        embeddings = OpenAIEmbeddings()

        if os.path.isdir(repo_path):
            loader = GitLoader(
                repo_path=repo_path,
                branch="main",
                file_filter=lambda file_path: any(
                    file_path.endswith(extension) for extension in FILE_PATH_WHITELIST
                ),
            )

        else:
            loader = GitLoader(
                clone_url=repo_url,
                repo_path=repo_path,
                branch="main",
                file_filter=lambda file_path: any(
                    file_path.endswith(extension) for extension in FILE_PATH_WHITELIST
                ),
            )

        docs = loader.load()

        Pinecone.from_documents(
            docs, embeddings, index_name="issue-sniffer", namespace=str(repo_id)
        )
