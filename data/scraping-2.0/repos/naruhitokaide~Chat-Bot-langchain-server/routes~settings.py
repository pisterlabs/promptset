from fastapi import APIRouter, UploadFile, Body
from pathlib import Path
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, YoutubeLoader
import os
import json

from utils import allowed_file, settings_check
from schemas.settings import BasicSettings

router = APIRouter(prefix="/api", tags=["settings"])

UPLOAD_FOLDER = './files'


@router.post('/upload/{num}')
async def upload(file: UploadFile, num: str):
    path = Path(UPLOAD_FOLDER) / file.filename

    if file and allowed_file(file.filename):
        path.write_bytes(await file.read())
        fileext = file.filename.rsplit('.', 1)[1].lower()
        if (fileext == 'pdf'):
            reader = PdfReader(path)
            raw_text = ''
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    raw_text += text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, length_function=len)
            texts = text_splitter.split_text(raw_text)
            embeddings = OpenAIEmbeddings()
            if os.path.exists(f"./store/{num}/index.faiss"):
                docsearch = FAISS.load_local(f"./store/{num}", embeddings)
                docsearch.add_texts(texts)
            else:
                docsearch = FAISS.from_texts(texts, embeddings)
            docsearch.save_local(f"./store/{num}")
        else:
            loader = TextLoader(path)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, length_function=len)
            split_docs = loader.load_and_split(text_splitter)
            embeddings = OpenAIEmbeddings()
            if os.path.exists(f"./store/{num}/index.faiss"):
                docsearch = FAISS.load_local(f"./store/{num}", embeddings)
                docsearch.add_documents(split_docs)
            else:
                docsearch = FAISS.from_documents(split_docs, embeddings)
            docsearch.save_local(f"./store/{num}")

        return {"state": "success"}
    return {"state": "error", "message": "Invalid file format"}


@router.post('/youtube/train/{num}')
async def train_youtube(num: int, url: str = Body(embed=True)):
    loader = YoutubeLoader.from_youtube_channel(url)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()
    if (os.path.exists(f"./store/{num}/index.faiss")):
        docsearch = FAISS.load_local(f"./store/{num}", embeddings)
        docsearch.add_documents(texts)
    else:
        docsearch = FAISS.from_documents(texts, embeddings)
    docsearch.save_local(f"./store/{num}")
    return {"state": "success"}


@router.post("/header-change/{num}")
async def header_change(num: str, item: BasicSettings):
    item_dict = item.dict()
    print(item_dict)
    settings_check(num)
    with open(f"./settings/{num}/settings.json") as f:
        data = json.load(f)
    data["title"] = item_dict["title"]
    data["model"] = item_dict["model"]
    with open(f"./settings/{num}/settings.json", "w") as f:
        json.dump(data, f)
    return {"status": "success"}


@router.post("/header-upload/{num}")
async def header_upload(file: UploadFile, num: str):
    settings_check(num)
    fileext = file.filename.rsplit('.', 1)[1].lower()
    path = Path(f"./settings/{num}") / f'header.{fileext}'
    path.write_bytes(await file.read())
    with open(f"./settings/{num}/settings.json") as f:
        data = json.load(f)
    data["header"] = f"header.{fileext}"
    with open(f"./settings/{num}/settings.json", "w") as f:
        json.dump(data, f)


@router.post("/botimg-upload/{num}")
async def botimg_upload(file: UploadFile, num: str):
    settings_check(num)
    fileext = file.filename.rsplit('.', 1)[1].lower()
    path = Path(f"./settings/{num}") / f'bot.{fileext}'
    path.write_bytes(await file.read())
    with open(f"./settings/{num}/settings.json") as f:
        data = json.load(f)
    data["bot"] = f"bot.{fileext}"
    with open(f"./settings/{num}/settings.json", "w") as f:
        json.dump(data, f)


@router.post("/userimg-upload/{num}")
async def userimg_upload(file: UploadFile, num: str):
    settings_check(num)
    fileext = file.filename.rsplit('.', 1)[1].lower()
    path = Path(f"./settings/{num}") / f'user.{fileext}'
    path.write_bytes(await file.read())
    with open(f"./settings/{num}/settings.json") as f:
        data = json.load(f)
    data["user"] = f"user.{fileext}"
    with open(f"./settings/{num}/settings.json", "w") as f:
        json.dump(data, f)


@router.get("/settings/{num}")
async def get_settings(num: str):
    settings_check(num)
    with open(f"./settings/{num}/settings.json") as f:
        data = json.load(f)
        return data
