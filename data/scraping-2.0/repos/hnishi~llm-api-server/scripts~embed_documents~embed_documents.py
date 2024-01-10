import argparse

from langchain.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient


def embed_document(document_path: str, qdrant_path: str, collection_name: str):
    embeddings = OpenAIEmbeddings()

    create_documents = True
    qdrant = None

    if create_documents:
        txt_docs = DirectoryLoader(
            document_path, glob="**/*.txt", loader_cls=TextLoader
        ).load()
        md_docs = DirectoryLoader(
            document_path, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
        ).load()
        mdx_docs = DirectoryLoader(
            document_path, glob="**/*.mdx", loader_cls=UnstructuredMarkdownLoader
        ).load()
        raw_docs = txt_docs + md_docs + mdx_docs

        # chunk_size のデフォルトは 4000
        # https://api.python.langchain.com/en/latest/_modules/langchain/text_splitter.html#CharacterTextSplitter.__init__
        text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
        # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(raw_docs)

        qdrant = Qdrant.from_documents(
            docs,
            embeddings,
            path=qdrant_path,
            collection_name=collection_name,
            # the collection is going to be reused if it already exists
            # https://python.langchain.com/docs/integrations/vectorstores/qdrant#recreating-the-collection
            force_recreate=True,
        )
    else:
        # https://github.com/qdrant/qdrant-client
        client = QdrantClient(path=qdrant_path)

        # https://api.python.langchain.com/en/latest/vectorstores/langchain.vectorstores.qdrant.Qdrant.html
        qdrant = Qdrant(client, collection_name=collection_name, embeddings=embeddings)

    query = "how to load markdown files to langchain?"
    found_docs = qdrant.similarity_search_with_score(query)

    # print(found_docs[0])
    document, score = found_docs[0]
    print(document)
    print(f"\nScore: {score}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--document-path",
        help="file path for document",
        type=str,
        default="./samples/state_of_the_union.txt",
    )
    parser.add_argument(
        "--qdrant-path",
        help="file path for qdrant on-disk storage",
        type=str,
        default="./local_qdrant",
    )
    parser.add_argument(
        "--collection-name",
        help="collection name for qdrant",
        type=str,
        default="my_documents",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    embed_document(args.document_path, args.qdrant_path, args.collection_name)
