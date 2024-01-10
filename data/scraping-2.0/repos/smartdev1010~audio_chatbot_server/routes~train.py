import os
from fastapi import APIRouter, HTTPException, Depends, status, Request, Body
from datetime import datetime
from flask import Flask, request, make_response
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/train", tags=["train"])

chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
docsearch = FAISS.from_documents(
    [Document(page_content="This is ZK-Rollup Crypto Info Data.\n\n")],
    OpenAIEmbeddings(),
)


def allowed_file(filename, ALLOWED_EXTENSIONS):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@router.post("/pdf", description="pdf train")
def upload():
    global reader, raw_text, texts, embeddings, docsearch

    if "file" not in request.files:
        return {"state": "error", "message": "No file part"}
    file = request.files["file"]
    api_key = request.form["token"]
    os.environ["OPENAI_API_KEY"] = api_key

    if file.filename == "":
        return {"state": "error", "message": "No selected file"}
    if file and allowed_file(file.filename, {"pdf"}):
        filename = secure_filename(file.filename)
        file.save(os.path.join("./traindata" + filename))
        reader = PdfReader(os.path.join("./traindata" + filename))
        raw_text = ""
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )
        texts = text_splitter.split_text(raw_text)

        docsearch = FAISS.from_texts(texts, OpenAIEmbeddings())
        dt = datetime.now()
        persona = dt.strftime("%Y%m%d%H%M%S")
        docsearch.save_local("./store/" + persona)

        return {"state": "success", "persona": persona}
    return {"state": "error", "message": "Invalid file format"}
