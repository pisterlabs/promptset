from langchain.vectorstores import Chroma

from constants import REPO_PATH


def generate_document_ids(documents):
    """
    current dcouments source is absolute path
    we need only the the relative path from the git repo
    """

    repo_folder_path = REPO_PATH + "/"

    document_ids = [
        doc.metadata["source"].replace(repo_folder_path, "") for doc in documents
    ]

    return document_ids


def create_codebase_embeddings(documents, embedding, persist_directory):
    """
    Remember we don't need to provide the collection name as we will use langchain default
    """
    return Chroma.from_documents(
        documents,
        embedding=embedding,
        ids=generate_document_ids(documents),
        persist_directory=persist_directory,
    )


def get_db(persist_directory, embedding):
    db = Chroma(
        embedding_function=embedding,
        persist_directory=persist_directory,
    )
    return db
