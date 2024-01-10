from pathlib import Path
from shutil import rmtree

from langchain.document_loaders import GitLoader

from lib.loaders.loader_utils import create_embeddings, chunk_docs


def load(repo, branch, file_filter=None):
    if is_git_url(repo):
        path = Path("tmp")
        path.mkdir(parents=True, exist_ok=True)
        docs = load_repo_url(path, repo, branch, file_filter)
        add_source_tags(docs)
        docs = chunk_docs(docs)
        create_embeddings(docs)
        rmtree(path, ignore_errors=True)
    else:
        docs = load_repo_path(repo, branch, file_filter)
        create_embeddings(docs)


def is_git_url(string: str):
    return (
            string.startswith("git@")
            or string.startswith("https://")
            or string.startswith("http://")
            or string.startswith("ssh://")
    ) and ".git" in string


def load_repo_url(path, url, branch, file_filter):
    try:
        return GitLoader(
            str(path),
            clone_url=url,
            branch=branch,
            file_filter=file_filter
        ).load()
    except Exception as e:
        print(f"could not load repo {url} / {branch}")
        return []


def add_source_tags(docs):
    for doc in docs:
        doc.metadata['source'] = doc.metadata['file_path']


def load_repo_path(path, branch, file_filter):
    try:
        return GitLoader(
            str(path),
            branch=branch,
            file_filter=file_filter
        ).load_and_split()
    except:
        print(f"could not load repo {path} / {branch}")
        return []
