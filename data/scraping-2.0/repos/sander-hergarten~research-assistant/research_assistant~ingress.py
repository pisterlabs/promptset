import os

from langchain.vectorstores import Qdrant
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
    
from .utils import config


documents = []
for file in os.listdir(config["assistant"]["ingress-location"]):
    documents.extend(PyPDFLoader(file).load_and_split())


model = SentenceTransformerEmbeddings(model=config["assistant"]['embedding-model'])


match config['assistant']['db_location']:
    case 'local':
        qdrant = Qdrant.from_documents(
            documents,
            model,
            location=config['assistant']['location'],
            collection_name='papers',
        )

    case 'remote':
        qdrant = Qdrant.from_documents(
            documents,
            model,
            url=config['assistant']['url'],
            prefer_grpc=True,
            collection_name='papers',
        )
    case 'memory':
        qdrant = Qdrant.from_documents(
            documents,
            model,
            location=":memory:",
            collection_name='papers',
        )

