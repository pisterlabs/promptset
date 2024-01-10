import git
import os
import re
from collections import defaultdict
from langchain.document_loaders import TextLoader, NotebookLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def clone_repo(github_url, repo_path, access_token):
    if os.path.exists(repo_path):
        return
    try:
        parts = re.split(r"(github\.com)", github_url)
        url_with_token = f"{parts[0]}{access_token}@{parts[1]}{parts[2]}"
        git.Repo.clone_from(url_with_token, repo_path)
    except:
        raise Exception("Invalid GitHub URL")


def split_files(repo_path):
    extension_freqs = defaultdict(int)
    document_chunks = []
    for dir_path, _, file_names in os.walk(repo_path):
        for file in file_names:
            file_path = os.path.join(dir_path, file)
            ext = os.path.splitext(file)[1]
            loader = TextLoader(file_path, encoding="utf-8")
            try:
                if ext == ".ipynb":
                    loader = NotebookLoader(
                        file_path,
                        include_outputs=True,
                        max_output_length=20,
                        remove_newline=True,
                    )
                document_chunks.extend(
                    loader.load_and_split(
                        text_splitter=RecursiveCharacterTextSplitter(chunk_size=250)
                    )
                )
                extension_freqs[ext] += 1
            except Exception as e:
                pass

    return (extension_freqs, document_chunks)
