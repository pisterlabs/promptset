from langchain.vectorstores import Chroma
from chromadb.config import Settings
from chromadb import PersistentClient
from langchain.embeddings import HuggingFaceInstructEmbeddings
from .models.skillfit_predictor import skillfit_predictor
from .models.storage import storage
import os


def load_embedding():
    return HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large",
        embed_instruction="Represent the document for retrieval: ",
        query_instruction="Represent the query for retrieval: ",
    )


def load_escodb(embedding):
    return Chroma(
        client=PersistentClient("./data/stores/esco_vectorstore"),
        embedding_function=embedding,
        client_settings=Settings(anonymized_telemetry=False),
    )


def load_dkzdb(embedding):
    return Chroma(
        client=PersistentClient("./data/stores/dkz_competencies_vectorstore"),
        embedding_function=embedding,
        client_settings=Settings(anonymized_telemetry=False),
    )


def load_skillfit_model():
    return skillfit_predictor()


def setup():
    embedding = load_embedding()
    skilldbs = {
        "ESCO": load_escodb(embedding),
        "DKZ": load_dkzdb(embedding),
    }
    skillfit_model = load_skillfit_model()

    db = storage()

    return embedding, skilldbs, skillfit_model, db
