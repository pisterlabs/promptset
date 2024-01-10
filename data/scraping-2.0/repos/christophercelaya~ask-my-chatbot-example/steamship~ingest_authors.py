import re
from pathlib import Path
from typing import Dict

import toml as toml
from langchain.document_loaders import PagedPDFSplitter
from steamship import File, Tag
from steamship import Steamship
from steamship_langchain.vectorstores import SteamshipVectorStore


def add_author_info(client: Steamship, index_name: str, author_info: Dict[str, str]):
    for file in File.query(client, tag_filter_query='all').files:
        file.delete()

    File.create(client, tags=[
        Tag(kind="AuthorTag", name=index_name, value=author_info)
    ])


def to_snake(author_name: str):
    return '_'.join(
        re.sub('([A-Z][a-z]+)', r' \1',
               re.sub('([A-Z]+)', r' \1',
                      author_name.replace('-', ' '))).split()).lower()


for folder in Path("uploads/authors").iterdir():
    author_name = folder.name
    index_name = to_snake(author_name)
    print(index_name)
    metadata = toml.load(folder / "metadata.toml")

    # connect to the workspace
    client = Steamship(workspace=index_name)

    # add_author_info(client=client, index_name=index_name, author_info={
    #     "authorName": author_name,
    #     **metadata
    # })

    doc_index = SteamshipVectorStore(client=client,
                                     index_name=index_name,
                                     embedding="text-embedding-ada-002")
    # doc_index.index.reset()

    for book in folder.iterdir():
        if book.suffix == ".pdf":
            print(f"\t{book}")
            loader = PagedPDFSplitter(str(book))
            pages = loader.load_and_split()

            doc_index.add_texts(
                texts=[page.page_content for page in pages],
                metadatas=[{**page.metadata, "source": book.name} for page in pages],
            )
