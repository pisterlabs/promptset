# Ejemplo de uso de un prompt que se conecta a Confluence y genera casos de prueba del requerimiento que se le asigne

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import pathlib
import sys

_parentdir = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(_parentdir))
print(_parentdir)

from scripts.config import Config

cfg = Config()

persist_directory = "examples/docs/chroma"
embeddings = OpenAIEmbeddings()
vectordb = Chroma(embedding_function=embeddings, persist_directory=persist_directory)

print(vectordb._collection.count())

question = "como esco fondos resuelve la funcionalidad de alquiler de títulos"
docs = vectordb.max_marginal_relevance_search(question, k=2, fetch_k=3)

print(docs[0].page_content)
