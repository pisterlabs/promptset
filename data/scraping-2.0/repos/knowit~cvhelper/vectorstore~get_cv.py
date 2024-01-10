from glob import glob
import os
import sys
from cvpartner_client import CVPartnerClient
import json
from langchain.vectorstores import Chroma
import chromadb
from split import CVSplitter
from langchain.embeddings import GPT4AllEmbeddings



def main():
    # Get token and download path from command line arguments
    token = os.environ["CVPARTNER_TOKEN"]
    download_path = sys.argv[1]

    # Create an instance of CVPartnerClient
    client = CVPartnerClient(token=token)

    # Download all CVs
    populate_vectordatabase(download_path)

def populate_vectordatabase(directory):
    for filename in os.listdir(directory):
        full_filename = os.path.join(directory, filename)
        print (full_filename)
        if os.path.isfile(full_filename):
            embed_and_insert(full_filename)


def split_all_cvs(path: str):
    splitter = CVSplitter(lang="no")
    for cv_path in glob(path):
        with open(cv_path) as f:
            cv = json.load(f)
            for item in splitter.get_all_splits(cv):
                yield item


def embed_and_insert(path: str):
    chroma_host = os.environ["CHROMA_HOST"]
    chroma_port = os.environ.get("CHROMA_PORT", "8000")
    embeddings = GPT4AllEmbeddings()
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    vectorstore = Chroma(
        collection_name="cv_no_clean",
        embedding_function=embeddings,
        client=chroma_client,
    )
    text_list = []
    meta_list = []
    for text, meta in split_all_cvs(path):
        text_list.append(text)
        meta_list.append(meta)

    vectorstore.add_texts(text_list, meta_list)

if __name__ == "__main__":
    main()
