"""This is the logic for ingesting Notion data into LangChain."""
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

from globals import Globals
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import faiss
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings,CohereEmbeddings
import pickle

#@Globals.stub.function(secret=modal.Secret.from_name(Globals.OPENAPI_SECRET), mounts=[modal.Mount.from_local_dir("./data", remote_path="./data")], memory=Globals.MEMORY, cpu=Globals.CORES, shared_volumes={Globals.CACHE_DIR: Globals.VOLUME})
# stub = modal.Stub(image=modal.Image.debian_slim().pip_install("openai","langchain","faiss-cpu","requests"))

# @stub.function(secret=modal.Secret.from_name(Globals["OPENAPI_SECRET"]), memory=Globals["MEMORY"], cpu=Globals["CORES"], shared_volumes={Globals["CACHE_DIR"]: Globals["VOLUME"]}, timeout=3600, interactive=True)
def ingest_all():
    print("Importing...")
    # Here we load in the data in the format that Notion exports it in.
    # ps = list(Path("/internal/dd/").glob("**/*.html"))
    # import zipfile
    # if len(ps) <= 0:
    #     print(f"Detected {len(ps)} records, getting zip now.")
    #     import requests
    #     import zipfile
    #     r = requests.get(Globals["DEVDOCS_ZIP"], stream = True) # create HTTP response object
    #     print("gotten")
    #     # send a HTTP request to the server and save
    #     # the HTTP response in a response object called r
    #     with open("/internal/devdocs.zip",'wb') as f:
    #         for chunk in r.iter_content(chunk_size=1024):
    #             if chunk:
    #                 f.write(chunk)
    #     print("extracting")
    ps = list(Path("./data/devdocs").glob("**/*.html"))
    print(f"Detected {len(ps)} records...")
    
    data = []
    sources = []
    for p in ps:
        try:
            with open(p) as f:
                data.append(f.read())
            sources.append(p)
        except Exception as e:
            print(f"An exception occurred during when loading file '{p}':", e)
    print("Memory load success.")
    print("Ingesting data...")


    # Here we split the documents, as needed, into smaller chunks.
    # We do this due to the context limits of the LLMs.
    text_splitter = CharacterTextSplitter(chunk_size=256, separator="\n")
    print('A')
    docs = []
    metadatas = []
    for i, d in enumerate(data):
        splits = text_splitter.split_text(d)
        docs.extend(splits)
        metadatas.extend([{"source": sources[i]}] * len(splits))

    print("E")
    # Here we create a vector store from the documents and save it to disk.
    store = FAISS.from_texts(docs, OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"]), metadatas=metadatas)
    # store = FAISS.from_texts(docs, CohereEmbeddings(cohere_api_key="p238LWMvLwKxbJj3W4V2aPoRKQACMBLEPbkCq31O", truncate="END"), metadatas=metadatas)
    print("F")

    faiss.write_index(store.index, Globals["INDEX"])
    print("G")

    store.index = None
    with open(Globals["FAISS_PKL"], "wb") as f:
        pickle.dump(store, f)
    print("Done!")

def main():
    print("HIIII")
    ingest_all()
    print("BYEEEE")

main()