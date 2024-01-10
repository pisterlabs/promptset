from flask import Flask,jsonify, render_template, flash, redirect, url_for, Markup, request
from flask_cors import CORS
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.vectorstores import FAISS #, Chroma
from langchain.llms import LlamaCpp #, GPT4All
import os
import glob
from typing import List
import requests
from huggingface_hub import hf_hub_download
from systemprompt import PROMPT

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
#from constants import CHROMA_SETTINGS

app = Flask(__name__)
CORS(app)

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
llm = None

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> Document:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]

    raise ValueError(f"Nicht unterstütztes Datei-Format: '{ext}'")


def load_documents(source_dir: str) -> List[Document]:
    # Loads all documents from source documents directory
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    return [load_single_document(file_path) for file_path in all_files]

@app.route('/ingest', methods=['GET'])
def ingest_data():
    # Load environment variables
    persist_directory = os.environ.get('PERSIST_DIRECTORY')
    source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
    embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')

    # Load documents and split in chunks
    print(f"Loading documents from {source_directory}")
    chunk_size = 500
    chunk_overlap = 50
    documents = load_documents(source_directory)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} documents from {source_directory}")
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} characters each)")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=embeddings_model_name,
        model_kwargs={'device': 'cpu'}
    )
    
    # Create vector store
    db = FAISS.from_documents(texts, embeddings)
    db.save_local('vectorstore/db_faiss')
    db = None
    return jsonify(response="Success")
    
@app.route('/get_answer', methods=['POST'])
def get_answer():
    query = request.json
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    vectordb = FAISS.load_local('vectorstore/db_faiss', embeddings)
    retriever = vectordb.as_retriever()
    if llm==None:
        return "Model not downloaded", 400  
    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs)
    if query!=None and query!="":
        res = qa(query)
        answer, docs = res['result'], res['source_documents']
        
        source_data =[]
        for document in docs:
             source_data.append({"name":document.metadata["source"]})

        return jsonify(query=query,answer=answer,source=source_data)

    return "Empty Query",400


@app.route('/upload_doc', methods=['POST'])
def upload_doc():
    
    if 'document' not in request.files:
        return jsonify(response="Kein Dokument gefunden"), 400
    
    document = request.files['document']
    if document.filename == '':
        return jsonify(response="Keine Datei ausgewählt"), 400

    filename = document.filename
    save_path = os.path.join('source_documents', filename)
    document.save(save_path)

    return jsonify(response="Document upload successful")

''' --Auskommentiert weil der UI Element Button "Download Model" entfernt wurde--
@app.route('/download_model', methods=['GET'])
def download_and_save():
    url = 'https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin'  # Specify the URL of the resource to download
    filename = 'ggml-gpt4all-j-v1.3-groovy.bin'  # Specify the name for the downloaded file
    models_folder = 'models'  # Specify the name of the folder inside the Flask app root

    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    response = requests.get(url,stream=True)
    total_size = int(response.headers.get('content-length', 0))
    bytes_downloaded = 0
    file_path = f'{models_folder}/{filename}'
    #if os.path.exists(file_path):
    #    return jsonify(response="Download completed")

    with open(file_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=4096):
            file.write(chunk)
            bytes_downloaded += len(chunk)
            progress = round((bytes_downloaded / total_size) * 100, 2)
            print(f'Download Progress: {progress}%')
    global llm
    callbacks = [StreamingStdOutCallbackHandler()]
    lm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
    #return jsonify(response="Download completed")
'''

def load_model():
    models_folder = 'models'  # Specify the name of the folder inside the Flask app root
    model_id="TheBloke/Llama-2-7B-Chat-GGML"
    model_basename = "llama-2-7b-chat.ggmlv3.q8_0.bin"
    model_path = hf_hub_download(repo_id=model_id, filename=model_basename)
    print(model_path)
    if os.path.exists(model_path):
        global llm
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        llm = LlamaCpp(model_path=model_path, n_ctx=2048, max_tokens=2048, temperature=0, repeat_penalty=1.15, callback_manager=callback_manager, verbose=True)

if __name__ == "__main__":
  load_model()
  print("LLM=", llm)
  app.run(host="0.0.0.0", port=1337, debug = False)
