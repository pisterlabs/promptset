import os
import uuid

from flask import Blueprint, jsonify, request
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from werkzeug.utils import secure_filename

from config import upload_folder as upload_folder
from flask_login import login_required


pdf_uploader = Blueprint("pdf_uploader", __name__)

ALLOWED_EXTENSIONS = {"pdf"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@pdf_uploader.route("/pdf-upload", methods=["POST"])
@login_required
async def pdf_upload():
    if "file" not in request.files:
        return jsonify(error="No file part"), 400
    file = request.files["file"]

    if file.filename == "":
        return jsonify(error="No selected file"), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        pdf_route = os.path.join(upload_folder, filename)
        file.save(pdf_route)

        loader = PyPDFLoader(file_path=pdf_route)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=30, separator="\n"
        )
        docs = text_splitter.split_documents(documents=documents)
        embeddings = OpenAIEmbeddings()
        uid = uuid.uuid4()
        dir_name = "storage/vector_indexes"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        index_name = f"storage/vector_indexes/faiss_index_react_{uid}"
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(index_name)

        new_vectorstore = FAISS.load_local(index_name, embeddings)
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever()
        )
        response = qa.run("Search a short title for the project")
        print(response)
        return (
            jsonify(
                status="File successfully uploaded",
                response=response,
                vector_index=index_name,
            ),
            200,
        )

    return jsonify(error="File not allowed"), 400
