from google.cloud import storage
from io import BytesIO
import os
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS



def download_from_gcs():

    load_dotenv()

    # Definer bucket- og blobnavn
    bucket_name = 'chatty1'
    blob_name = 'COAX_web_content.pkl'

    # Setter nøkkelbanen for Google Cloud-autentisering
    key_file = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if key_file is None:
        raise Exception("Google Cloud-nøkkelfilen er ikke definert i miljøvariablene")


    # Autentiser ved hjelp av nøkkelfilen
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_file

    # Opprett en klient
    storage_client = storage.Client()

    # Få tak i bøtta
    bucket = storage_client.bucket(bucket_name)

    # Hent blob (filen)
    blob = bucket.blob(blob_name)

    # Opprett en BytesIO-strøm og last ned dataen direkte til denne
    data_stream = BytesIO()
    blob.download_to_file(data_stream)
    data_stream.seek(0)  # Tilbakestill pekeren til starten av strømmen

    # Deserialiser dataen til et FAISS VectorStore-objekt
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=data_stream.getvalue())

    return vector_store

