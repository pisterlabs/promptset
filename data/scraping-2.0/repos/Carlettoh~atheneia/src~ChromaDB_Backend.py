from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyMuPDFLoader, AmazonTextractPDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import os
from chromadb.utils import embedding_functions

embeddings = OpenAIEmbeddings()


# Crea una instancia del cliente ChromaDB
chroma_client = chromadb.Client()



feynman_lectures_path = '/home/pi/Documents/academIA/docs/Serway.pdf'
loader = PyMuPDFLoader(feynman_lectures_path)
documents = loader.load()

# Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
texts = text_splitter.split_documents(documents)

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ['OPENAI_API_KEY'],
                model_name="text-embedding-ada-002"
            )



embeddings_result = embeddings.embed(texts)



# Define los datos que deseas insertar en la base de datos
datos_pdf = {
    "titulo": feynman_lectures_path,
    "contenido": "Texto extraido del PDF",
    "autor": "Autor del PDF",
    "fecha_publicacion": "Fecha de publicacion del PDF",
    "embeddings": embeddings_result,  # Lista de embeddings del PDF
}

# Utiliza el metodo insert_document() para insertar los datos en la base de datos
embeddings_pdf = chroma_client.insert_document(datos_pdf)