import os

import pathspec as pathspec
from langchain.document_loaders import TextLoader

from lib.loaders.hash_cache import check_path, update_file_hash
from lib.loaders.loader_utils import create_embeddings, get_ignore_patterns


def load(source):
    print(os.getcwd())
    if os.path.isfile(source):
        docs = load_file(source)
    elif os.path.isdir(source):
        docs = load_directory(source)
    else:
        print("file not found")
        return
    create_embeddings(docs)


""" Returns true if file hash has been changed since last load call. """


def load_directory(directory):
    ignore_patterns = get_ignore_patterns(os.path.join(directory, ".gitignore"))
    spec = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, ignore_patterns)
    docs = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, directory)
            if not spec.match_file(relative_path) and check_path(file_path):
                docs.extend(load_file(file_path))
    print(f'loaded {len(docs)} documents')
    return docs


def load_file(file_path):
    try:
        docs = TextLoader(file_path, encoding='utf-8').load_and_split()
        update_file_hash(file_path)
        return docs
    except Exception as e:
        print(f"failed to load file {file_path}")
        return []
