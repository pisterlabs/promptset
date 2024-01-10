from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader
import chromadb
import os


def make_collection():
    # setup Chroma in-memory, for easy prototyping. Can add persistence easily!
    client = chromadb.Client()

    # Create collection. get_collection, get_or_create_collection, delete_collection also available!
    collection = client.create_collection("jobs")

    folder_path = "./jobs"
    documents = []
    ids = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                content = file.read()
                documents.append(content)
        ids.append(str(len(documents)))
    # Add docs to the collection. Can also update and delete. Row-based API coming soon!
    collection.add(
        documents=documents,  # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
        # metadatas=[{"source": "notion"}, {"source": "google-docs"}], # filter on these!
        ids=ids,  # unique for each doc
    )
    return collection


def query_collection(collection, query):
    # Query/search 2 most similar results. You can also .get by id
    results = collection.query(
        query_texts=[query],
        n_results=3,
    )
    return results


if __name__ == "__main__":
    collection = make_collection()
    results = query_collection(collection, "data science")
    print(results)
