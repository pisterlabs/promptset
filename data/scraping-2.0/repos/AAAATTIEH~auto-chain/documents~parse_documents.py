
from utils.helpers import saveTemp
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.blob_loaders.file_system import FileSystemBlobLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders.parsers.audio import OpenAIWhisperParser
import os
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.document_loaders import Docx2txtLoader
from models.tools.image_caption import ImageCaptionTool
import streamlit as st

def parse_pdf(file):
    tmp_file = saveTemp(file)
    tmp_file_path = tmp_file["file"]
    loader = PyPDFLoader(tmp_file_path)
    chunks = loader.load_and_split()
    return tmp_file_path,chunks
def path_to_blob(file_path):
    with open(file_path, "rb") as file:
        blob = file.read()
    return blob
def parse_csv(file):
    tmp_file = saveTemp(file,f"dataset/process/{st.session_state.user_id}/{st.session_state.session_id}/input/tables")
    tmp_file_path = tmp_file["file"]

    return tmp_file_path
def parse_pptx(file):
    tmp_file = saveTemp(file)
    tmp_file_path = tmp_file["file"]
    loader = UnstructuredPowerPointLoader(file_path=tmp_file_path)
    data = loader.load()
    return tmp_file_path,data
def parse_docx(file):
    tmp_file = saveTemp(file)
    tmp_file_path = tmp_file["file"]
    loader = Docx2txtLoader(file_path=tmp_file_path)
    data = loader.load_with_images(tmp_file["name"])
    
    
    return tmp_file_path,data
def parse_txt(file):
    tmp_file = saveTemp(file)
    tmp_file_path = tmp_file["file"]
    loader = TextLoader(file_path=tmp_file_path,autodetect_encoding=True)
   
    data = loader.load()
    
    return tmp_file_path,data
def parse_image(file):
    tmp_file = saveTemp(file,f"dataset/process/{st.session_state.user_id}/{st.session_state.session_id}/input/images")
    tmp_file_path = tmp_file["file"]
    metadata = ImageCaptionTool().run(tmp_file_path)
    try:
        loader = UnstructuredImageLoader(file_path=tmp_file_path)
        data = loader.load()
    except:
        data = []
    return tmp_file_path,data,metadata
     

def parse_audio(file):
    tmp_file = saveTemp(file)
    loader = GenericLoader(FileSystemBlobLoader(tmp_file),OpenAIWhisperParser(api_key=os.getenv("OPENAI_API_KEY")))
    data = loader.load()

    return data

def parse_links(file):
    loader = WebBaseLoader("https://www.brookings.edu/articles/how-artificial-intelligence-is-transforming-the-world/")
    data = loader.load()
    return data
