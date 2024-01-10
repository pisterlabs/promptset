import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from pdfminer.high_level import extract_pages
from pdfminer.high_level import extract_text
import os
import pytesseract

### PATHS ###
# Para instalar teseract deberas hacer los pasos que se realizan en el siguiente link: https://www.youtube.com/watch?v=3Q1gTDXzGnU&t=12s
# Al finalizar utiliza el comando que se encuenta en \Layla_Sphere\zona de pruebas\Conseguir_Rutas.ipynb para verificar
# que la ruta esta correctamente implementada, al finalizar, solo pega la ruta en el codigo que se encuentra abajo

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

### CARGAR DOCUMENTOS ###


def load_document(file, file2, file3):
    import os

    file = "C:/Users/antonio.gutierrez/Documents/Layla_Sphere/database/CV S4B Completo_Español_06.2023.pptx"
    file2 = "C:/Users/antonio.gutierrez/Documents/Layla_Sphere/database/Guia Rapida - SOL-KAT-001 Solicitud de Requisición y OC.pptx"
    file3 = "C:/Users/antonio.gutierrez/Documents/Layla_Sphere/database/M-SGI-001 Manual de Sistema de Gestion Integral_ v11.pdf"

    name, extension = os.path.splitext(file, file2, file3)

    if extension == ".pdf":
        from langchain.document_loaders import PyPDFLoader

        print(f"Loading{file3}")
        loader = PyPDFLoader(file3)

    elif extension == ".pptx":
        from langchain.document_loaders import UnstructuredPowerPointLoader

        print(f"Loading {file}")
        loader = UnstructuredPowerPointLoader(file, file2)

    else:
        print("Documento no soportado")
        return None

    data = loader.load()

    print(docs[0].page_content)