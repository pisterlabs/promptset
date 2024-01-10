from langchain.vectorstores import FAISS
from src.ml.embeddings import get_embeddings


def init_vector_db(vector_db_path="./vector_db/"):
    hf_embeddings = get_embeddings()

    db_jobs = FAISS.load_local(vector_db_path + "db_jobs", hf_embeddings)
    db_resume = FAISS.load_local(vector_db_path + "db_resume", hf_embeddings)

    return db_jobs, db_resume
