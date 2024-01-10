import os
import argparse
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake

from constants import *

def index_files(lake_name, files_dir, overwrite=False):
    embeddings = OpenAIEmbeddings(disallowed_special=())

    docs = []
    for dirpath, dirnames, filenames in os.walk(files_dir):
        for file in filenames:
            try: 
                loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                docs.extend(loader.load_and_split())
            except Exception as e: 
                print(f"Unable to load {file}")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)

    db = DeepLake(dataset_path=f"hub://{username_activeloop}/{lake_name}", embedding_function=embeddings, overwrite=overwrite)
    db.add_documents(texts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conduct indexing of data into DeepLake VectorStore.')
    parser.add_argument("--lake_name", "-ln", type=str, help="Name of the lake you wish to save this tensor index set into")
    parser.add_argument("--files_dir", "-fd", type=str, help="Directory for the files that to be indexed")
    parser.add_argument("--overwrite", "-ow", action="store_true", help="Flag for whether to overwrite existing indices in lake")

    parser.set_defaults(files_dir="./training")
    parser.set_defaults(overwrite=False)

    args = parser.parse_args()

    outp = index_files(args.lake_name, args.files_dir, args.overwrite)
