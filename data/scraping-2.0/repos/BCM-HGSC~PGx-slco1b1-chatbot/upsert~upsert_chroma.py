# -*- coding:utf-8 -*-
# Created by liwenw at 6/12/23

import os
import openai
from langchain.document_loaders import CSVLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import chromadb

from omegaconf import OmegaConf
import argparse
from source_metadata_mapping import pick_metadata


def create_parser():
    parser = argparse.ArgumentParser(description='demo how to use ai embeddings to chat.')
    parser.add_argument("-y", "--yaml", dest="yamlfile",
                        help="Yaml file for project", metavar="YAML")
    return parser


def upsert_csv(collection, filename, data_dir, i):
    metadata = pick_metadata(filename)
    csv_loader = CSVLoader(os.path.join(data_dir, filename))
    pages = csv_loader.load_and_split()
    for page in pages:
        # print(page.metadata)
        collection.upsert(
            documents=page.page_content,
            metadatas=[metadata],
            ids=[str(i)]
        )
        i += 1

    return i

def upsert_txt(collection, filename, data_dir, i):
    metadata = pick_metadata(filename)
    file = open(os.path.join(data_dir, filename), 'r')
    lines = file.readlines()
    for line in lines:
        # print(page.metadata)
        collection.upsert(
            documents=line,
            metadatas=[metadata],
            ids=[str(i)]
        )
        i += 1

    return i


def upsert_pdf(collection, filename, data_dir, i, chunk_size, chunk_overlap):
    metadata = pick_metadata(filename)
    pdf_loader = PyPDFLoader(os.path.join(data_dir, filename))
    data = pdf_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(data)
    for text in texts:
        collection.upsert(
            documents=text.page_content,
            metadatas=[metadata],
            ids=[str(i)]
        )
        i += 1

    return i

def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.yamlfile is None:
        parser.print_help()
        exit()

    yamlfile = args.yamlfile
    config = OmegaConf.load(yamlfile)
    data_dirs = config.data.directory
    chunk_size = config.parse_pdf.chunk_size
    chunk_overlap = config.parse_pdf.chunk_overlap

    # Create a new Chroma client with persistence enabled.
    persist_directory = config.chromadb.persist_directory
    chroma_db_impl = config.chromadb.chroma_db_impl
    chroma_client = chromadb.Client(Settings(
        chroma_db_impl=chroma_db_impl,
        persist_directory=persist_directory
    ))
    # Start from scratch
    chroma_client.reset()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        openai_api_key = config.openai.api_key

    # openai.api_key = config.openai.api_key
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=openai_api_key,
                    model_name=config.openai.embedding_model_name,
                )

    collection_name = config.chromadb.collection_name
    collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=openai_ef)
    i=collection.count()

    for data_dir in data_dirs:
        print(f"Ingest files at {data_dir}")
        for filename in os.listdir(data_dir):
            if filename.endswith(".pdf") and os.path.isfile(os.path.join(data_dir, filename)):
                print(f"Upserting {filename}")
                i = upsert_pdf(collection, filename, data_dir, i, chunk_size, chunk_overlap)
            elif filename.endswith(".csv") and os.path.isfile(os.path.join(data_dir, filename)):
                print(f"Upserting {filename}")
                i = upsert_txt(collection, filename, data_dir, i)
            else:
                continue


if __name__ == "__main__":
    main()