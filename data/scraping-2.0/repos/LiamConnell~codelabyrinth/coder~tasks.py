from datetime import datetime

from coder import vectorstore, doc_loaders, utils
from coder.db import Database


def ingest_directory(path, metadata=None, name=None):
    name = name or utils.get_git_hash()

    v = vectorstore.VectorStore()
    docs = doc_loaders.load_code_files(path)

    metadata = metadata or {}
    metadata.update({"codebase": path, "timestamp": str(datetime.now())})
    v.create_collection(name, metadata)
    v.add_docs(name, docs)


def ingest_website(base_url, metadata=None, name=None):
    name = name or base_url

    v = vectorstore.VectorStore()
    docs = doc_loaders.load_docs_website(base_url)

    metadata = metadata or {}
    metadata.update({"base_url": base_url, "timestamp": str(datetime.now())})
    v.create_collection(name, metadata)
    v.add_docs(name, docs)


def ingest_gee_docs_website(base_url, metadata=None, name=None):
    name = name or base_url

    v = vectorstore.VectorStore()
    docs = doc_loaders.load_gee_website()

    metadata = metadata or {}
    metadata.update({"base_url": base_url, "timestamp": str(datetime.now())})
    v.create_collection(name, metadata)
    v.add_docs(name, docs)

def ingest_github_repo(owner, repo, path="", file_types=None, metadata=None, name=None):
    name = name or f"{owner}/{repo}/{path}"
    v = vectorstore.VectorStore()
    docs = doc_loaders.load_github_repo(owner, repo, path, file_types)

    metadata = metadata or {}
    metadata.update({"owner": owner, "repo": repo, "path": path,
                             "file_types": file_types, "timestamp": str(datetime.now())})
    v.create_collection(name, metadata)
    v.add_docs(name, docs)


def refresh_collection(collection_name: str):
    db = Database()
    v = vectorstore.VectorStore()
    metadata = db.fetch_query(f"select cmetadata from langchain_pg_collection where name='{collection_name}'")[0][0]
    if 'codebase' in metadata:
        v.delete_collection(collection_name)
        ingest_directory(metadata['codebase'], metadata, collection_name)
    else:
        raise NotImplementedError


