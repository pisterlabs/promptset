# Langchain wraps the Milvus client and provides a few convenience methods for working with documents.
# It can split documents into chunks, embed them, and store them in Milvus.

import os
import argparse

from langchain.document_loaders import TextLoader, GitLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import SupabaseVectorStore
from langchain.document_loaders import UnstructuredMarkdownLoader, DirectoryLoader
from langchain.document_loaders.text import TextLoader

from git import Repo

from supabase.client import Client, create_client


def load_documents(file_path, encoding='utf8', file_type='text'):
    if file_type == 'markdown':
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding=encoding)
    return loader.load()


def load_documents_from_github(github_url, file_type, collection_name):
    repoPath = "./docs/" + collection_name
    githubUrl = github_url
    # check if the repo exists
    if os.path.exists(repoPath):
        print("Repo already exists")
        githubUrl = None

    loader = GitLoader(
        clone_url=githubUrl,
        repo_path=repoPath,
        branch="main",
        file_filter=lambda file_path: file_path.endswith(
            ".{}".format(file_type)),
    )

    return loader.load()


def load_documents_from_directory(file_type, collection_name):
    path = "./docs/" + collection_name
    loader = DirectoryLoader(
        path=path,
        glob="**/*.{}".format(file_type),
        loader_cls=TextLoader
    )

    return loader.load()


def clone_from_github(github_url, collection_name):
    path = "./docs/" + collection_name
    if os.path.exists(path):
        print("Repo already exists")
        return

    repo = Repo.clone_from(
        github_url, to_path=path
    )
    repo.git.checkout("main")


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)


def index_documents(supabaseVectorStore, docs):
    # Index the documents using the provided Milvus instance
    supabaseVectorStore.add_documents(docs)


def main(input_dir, encoding, chunk_size, chunk_overlap, supabase_url, supabase_service_key, file_type, table_name, query_name, github_url):
    embeddings = OpenAIEmbeddings()

    supabase: Client = create_client(supabase_url, supabase_service_key)
    # Create the VectorStore
    supabaseVectorStore = SupabaseVectorStore(
        supabase,
        embeddings,
        table_name,
        query_name,
    )

    documents = []
    if input_dir is not None:
        # Iterate through all the files in the input directory and process each one
        for file in os.listdir(input_dir):
            file_path = os.path.join(input_dir, file)
            if os.path.isfile(file_path):
                print(f"Processing {file_path}...")
                loadedDocuments = load_documents(
                    file_path, encoding, file_type)
                # concat two lists
                documents = documents + loadedDocuments
        docs = split_documents(documents, chunk_size, chunk_overlap)

        index_documents(supabaseVectorStore, docs)
        print(f"Indexed {len(docs)} chunks from {input_dir}.")

    if github_url is not None:
        documents = load_documents_from_directory(file_type, collection_name)
        docs = split_documents(documents, chunk_size, chunk_overlap)

        index_documents(supabaseVectorStore, docs)
        print(f"Indexed {len(docs)} chunks from {github_url}.")

    print("Done!")


# python src/index_documents.py --input_dir /path/to/your/documents --file_type markdown --collection_name my_collection
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index documents for Question Answering over Documents application.")
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Path to the directory containing documents to be indexed.')
    parser.add_argument('--encoding', type=str, default='utf8',
                        help='Encoding of the input documents.')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='Size of the chunks to split documents into.')
    parser.add_argument('--chunk_overlap', type=int, default=0,
                        help='Number of overlapping characters between consecutive chunks.')
    parser.add_argument('--supabase_url', type=str, required=True,
                        help='Supabase url.')
    parser.add_argument('--supabase_service_key', type=str, required=True,
                        help='Supabase service key.')
    parser.add_argument('--file_type', type=str, default="text", choices=[
                        "text", "markdown", "adoc"], help='Type of the input files (text or markdown).')
    parser.add_argument('--table_name', type=str, required=True,
                        help='Name of the table to index the documents into.')
    parser.add_argument('--query_name', type=str, required=True,
                        help='The function query name.')
    parser.add_argument('--github_url', type=str, default=None,
                        help='URL of the file to download from GitHub (raw content URL).')

    args = parser.parse_args()

    main(args.input_dir, args.encoding, args.chunk_size, args.chunk_overlap,
         args.supabase_url, args.supabase_service_key, args.file_type, args.table_name, args.query_name, args.github_url)
