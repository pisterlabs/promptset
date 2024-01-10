import os
import chromadb
import uuid
import logging
import json

from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings


from flask import Flask, request
from werkzeug.utils import secure_filename

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

CHROMA_CLIENT = chromadb.Client(Settings(chroma_api_impl="rest",
                                        chroma_server_host="localhost",
                                        chroma_server_http_port="8000",
                                        anonymized_telemetry=False
                                    ))

FOLDER = f"{os.getcwd()}/upload"
EMBEDDING_FUNCTION = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

if not os.path.exists(FOLDER):
    logging.info("Creating upload folder!")
    os.makedirs(FOLDER)


def upload_chromadb(collection, file_paths):
    try:
        loader = DirectoryLoader(FOLDER, glob="*.*")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        for doc in docs:
            fid = f"{str(uuid.uuid4())}"
            collection.add(ids=[fid], metadatas=doc.metadata, documents=doc.page_content)

        for file_path in file_paths:
            os.remove(file_path)

    except Exception as ex:
        raise ex


@app.route('/health', methods = ['GET'])
def health_check():
    check = CHROMA_CLIENT.heartbeat()
    if check:
        return {
            "service status": "Service is healthy!",
            "chromadb status": check,
        }, 200

    return {
        "service status": "Service is down :(",
        "chromaDB status": "ChromaDB is not alive :("
    }, 404


@app.route("/reset", methods = ['GET'])
def reset():
    try:
        CHROMA_CLIENT.reset()
        return {
            "message": "Reset successful!"
        }, 200
    except Exception as ex:
        return {
            "message": "Failed to reset :("
        }, 404


@app.route('/upload', methods = ['POST'])
def upload_file():

    request_form = request.form.get("json")
    json_body = json.loads(request_form)

    collection_name = json_body['collection']
    upload_files = request.files.getlist("file")

    file_paths = []

    try:
        for upload_file in upload_files:
            file_path = f"{FOLDER}/{secure_filename(upload_file.filename)}"
            file_paths.append(file_path)
            upload_file.save(file_path)

        collection = CHROMA_CLIENT.get_or_create_collection(collection_name)

        upload_chromadb(collection, file_paths)

        return {
            "message": "File uploaded successful",
            "collection count": collection.count()
        }, 200
    except Exception as ex:
        print(f"Error: {str(ex)}")
        return {
            "message" : "File uploaded not successful :(",
            "error": str(ex)
        }, 404


@app.route("/search", methods=['POST'])
def search():

    request_body = request.json
    query = request_body['query']
    collection_name = request_body['collection']

    db = Chroma(client=CHROMA_CLIENT, collection_name=collection_name, embedding_function=EMBEDDING_FUNCTION)
    docs = db.similarity_search(query)

    result = ""
    sources = set()

    for doc in docs:
        sources.add(doc.metadata['source'].split("/")[-1])
        result += doc.page_content

    return {
        "answer": result,
        "sources": list(sources)
    }, 200


if __name__ == '__main__':
   app.run(port=5002, debug=True)

# gunicorn --bind :5002 --workers 2 --threads 2 --timeout 0 chroma_server:app
