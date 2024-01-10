import json
import os
from uuid import uuid4

from fastapi import HTTPException
from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

json_path = os.path.join(os.path.dirname(__file__), "documents.json")

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.environ["hf_token"],
    model_name="sentence-transformers/all-MiniLM-l6-v2",
)


def get_documents_from_json():
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def add_doc_to_json(document_id, document_name):
    documents = get_documents_from_json()
    documents[document_id] = document_name
    with open(json_path, "w") as f:
        json.dump(documents, f, indent=4)


def delete_doc_from_json(document_id):
    documents = get_documents_from_json()
    del documents[document_id]
    with open(json_path, "w") as f:
        json.dump(documents, f, indent=4)


def store_to_df(store: FAISS):
    # TODO: Implement this
    v_dict = store.docstore._dict
    data_rows = []
    for k in v_dict.keys():
        doc_name = v_dict[k]


def add_to_vectorDB(docs):
    chunks = text_splitter.split_documents(docs)
    if os.path.exists("FAISS"):
        db = FAISS.load_local("FAISS", embeddings=embeddings)
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(chunks, embeddings)
    db.save_local("FAISS")


def load_doc(files: list):
    documents = []
    for file in files:
        filepath = os.path.join("tmp", file.filename)
        extension = file.filename.split(".")[-1]
        match extension:
            case "pdf":
                loader = PyPDFLoader(filepath)
            case "txt":
                loader = UnstructuredFileLoader(filepath)
            case _:
                raise HTTPException(status_code=415, detail="Unkown Filetype")
        data = loader.load()
        doc_id = str(uuid4())
        for doc in data:
            doc.metadata["doc_id"] = doc_id
            documents.append(doc)
        add_doc_to_json(doc_id, file.filename)
    return documents
