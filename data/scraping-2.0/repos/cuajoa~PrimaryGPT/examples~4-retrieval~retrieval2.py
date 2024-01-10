# Ejemplo de uso de un prompt que se conecta a Confluence y genera casos de prueba del requerimiento que se le asigne

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

import pathlib
import sys

_parentdir = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(_parentdir))
print(_parentdir)

from scripts.config import Config

cfg = Config()

persist_directory = "examples/docs/chroma"

metadata_field_info = [
    AttributeInfo(
        name="title",
        description="La lectura del chung es desde o debería ser de uno de los documentos de conflunece",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="La página de la lectura",
        type="integer",
    )
]

document_content_description = "Documento de confluence"
llm = OpenAI(temperature=0)


embeddings = OpenAIEmbeddings()
vectordb = Chroma(embedding_function=embeddings, persist_directory=persist_directory)

retriever = SelfQueryRetriever.from_llm(llm, vectordb,document_content_description, metadata_field_info, verbose=True)


question = "como esco fondos resuelve la funcionalidad de alquiler de títulos"
docs = retriever.get_relevant_documents(question)

print(docs[0].page_content)
