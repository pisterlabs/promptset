from langchain.vectorstores import Chroma

from database.data_loader import esops_documents
from model.model_loader import embeddings

persist_docs="chroma"
vector_db=Chroma.from_documents(
    documents=esops_documents,
    embedding=embeddings,
    persist_directory=persist_docs
)