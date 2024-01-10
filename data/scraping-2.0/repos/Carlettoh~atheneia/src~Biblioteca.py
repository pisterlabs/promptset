from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

import os
# from chromadb.utils import embedding_functions

class Biblioteca:

    def __init__(self):
        self.list_of_pdf_files = ['serway_7ed_vol_2', 'Tipler', 'Purcell', 'David_Griffiths', 'em_cambridge']
        self.path_pdf_files    = '/home/pi/Documents/academIA/docs/'

        # Configurar el splitter
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                            chunk_overlap=10)

        self.documents = []


    def load_files(self):
        for iter_pdf in self.list_of_pdf_files:
            print(iter_pdf)
            loader = PyPDFLoader(f'{self.path_pdf_files}{iter_pdf}.pdf')
            # Cada pagina es un documento
            self.documents.extend(loader.load())

    def load_one_file(self, name_file):
        loader = PyMuPDFLoader(f'{self.path_pdf_files}{name_file}.pdf')
        return loader.load()

    def split_text(self):
        print(len(self.documents))
        if len(self.documents)>0:
            return self.text_splitter.split_documents(self.documents)
        else:
            return None
class BiblioChroma(Biblioteca):
    def __init__(self):
        # Crea una instancia del cliente ChromaDB
        super().__init__()

    def init_db(self):
        self.db = Chroma.from_documents(self.split_text(),
                                        OpenAIEmbeddings(),
                                        persist_directory="/home/pi/Documents/academIA/data",
                                        )
        # self.conversations = self.db.get_or_create_collection("Conversations")

    def get_retriever(self):
        return self._client.as_retriever()

    def insert(self, document):
        self._client.insert(document)

if __name__ == "__main__":
    biblio = BiblioChroma()
    biblio.load_files()
    biblio.init_db()









# feynman_lectures_path = '/home/pi/Documents/academIA/docs/Serway.pdf'
# loader = PyMuPDFLoader(feynman_lectures_path)
# documents = loader.load()
#
# # Splitter
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
#                                                chunk_overlap=150)
# texts = text_splitter.split_documents(documents)
#
#
#
#
# # Define los datos que deseas insertar en la base de datos
# datos_pdf = {
#     "titulo": feynman_lectures_path,
#     "contenido": "Texto extraido del PDF",
#     "autor": "Autor del PDF",
#     "fecha_publicacion": "Fecha de publicacion del PDF",
#     "embeddings": embeddings_result,  # Lista de embeddings del PDF
# }
#
# # Utiliza el metodo insert_document() para insertar los datos en la base de datos
# embeddings_pdf = chroma_client.insert_document(datos_pdf)