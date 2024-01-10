from typing import List
from robocorp.tasks import task
from robocorp import vault, storage

from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma

from loaders.RPA import RPALoader
from loaders.Robo import RoboLoader
from loaders.Portal import PortalLoader


@task
def prepare_docs():
    """Ingest data from various sources and create embeddings to the
    vector db."""

    # Get credentials from Robocorp Control Room Vault
    openai_secrets = vault.get_secret("OpenAI")

    # Get configuration data from Robocorp Control Room Asset Storage
    config = storage.get_json("rag_loader_config")

    docs: List[Document] = []

    # Load RPA Framework from a documentation JSON file
    rpaLoader = RPALoader(
        url=config["rpa"]["url"],
        black_list=config["rpa"]["black_list"],
    )
    rpaDocs = rpaLoader.load()
    docs.extend(rpaDocs)

    # Load Robo Framework docs from Git repo and markdown files.
    roboLoader = RoboLoader(
        repo_url=config["robo"]["url"],
        white_list=config["robo"]["white_list"],
    )
    roboDocs = roboLoader.load()
    docs.extend(roboDocs)

    # Load Portal robot examples based on configuration file and git repos.
    portalLoader = PortalLoader(url=config["portal"]["url"])
    portalDocs = portalLoader.load()
    docs.extend(portalDocs)

    create_embeddings(docs, openai_secrets)

def create_embeddings(documents, openai_secrets):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_secrets["key"])

    # TODO: Use a cloud hosted vectordb as a showcase.
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="ROBOCORP_DOCS_TEST"
    )

    return db