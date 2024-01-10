import os
from glob import glob
import argparse

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from config import *


class Create:
    def __init__(self, split_chunk_size, split_chunk_overlap, embedding_chunk_size):
        """
        Path to data, vector DB and model to be usd are in config.py.
        They can be added in here for better class design.

        :param split_chunk_size:
        :param split_chunk_overlap:
        :param embedding_chunk_size:
        """
        self.split_chunk_size = split_chunk_size if split_chunk_size is not None else 2500
        self.split_chunk_overlap = split_chunk_overlap if split_chunk_overlap is not None else 250
        self.embedding_chunk_size = embedding_chunk_size if embedding_chunk_size is not None else 16

    def create(self):
        try:
            print("Starting Vector DB creation...")
            print(f"Splitting Chunk Size {self.split_chunk_size}")
            print(f"Splitting Chunk Overlap Size {self.split_chunk_overlap}")
            print(f"Splitting Embedding Chunk Size {self.embedding_chunk_size}")
            docs = []
            for each_file in glob(DATA):
                loader = PyPDFLoader(each_file)
                docs.extend(loader.load())
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.split_chunk_size,
                                                           chunk_overlap=self.split_chunk_overlap)
            splits = text_splitter.split_documents(docs)
            print(f"Total number of splits: {len(splits)}")
            embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'],
                                          model=EMBEDDING_MODEL,
                                          chunk_size=self.embedding_chunk_size)
            vectordb = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=VECDB_DIR)
            print("Vector DB is successfully created!")
        except Exception as e:
            print("There was an error while creating the vector DB.")
            print(str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-splitChunkSize", type=int, required=False)
    parser.add_argument("-splitChunkOverlap", type=int, required=False)
    parser.add_argument("-embChunkSize", type=int, required=False)
    args = parser.parse_args()
    create_vecdb = Create(args.splitChunkSize, args.splitChunkOverlap, args.embChunkSize)
    create_vecdb.create()



