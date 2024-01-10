import sys
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from dotenv import load_dotenv

sys.path.append("D:\MestradoUnifei\desafioXP\Desafio-tecnico-xp")

load_dotenv()

from src.patterns.database import *

db = DatabaseFactory().create_database(
    database_type="chroma",
    embeddings=OpenAIEmbeddings(),
    persist_directory="Desafio-tecnico-xp\src\data\chroma_store",
)

# print(db.get_vectors())

document_content_description = "Curriculos e documentos de identificação"

metadata_field_info = [
    AttributeInfo(
        name="filename",
        description="Nome do arquivo",
        is_searchable=True,
        is_facetable=True,
        is_filterable=True,
        type="string",
    ),
    AttributeInfo(
        name="content",
        description="Conteúdo do arquivo",
        is_searchable=True,
        is_facetable=True,
        is_filterable=True,
        type="string",
    ),
    AttributeInfo(
        name="metadata",
        description="Metadados do arquivo",
        is_searchable=True,
        is_facetable=True,
        is_filterable=True,
        type="string",
    ),
]

retriever = SelfQueryRetriever.from_llm(
    llm=OpenAI(temperature=0.3),
    vectorstore=db.get_vector_store(),
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
)

docs_int = db.get_vector_store().similarity_search("Jedi")


sources = []

for doc in docs_int:
    sources.append(doc.metadata["source"])

print(sources)
print(len(sources))

# print("-" * 15)

# newDB = Chroma.from_documents(
#     documents=docs_int, embedding=OpenAIEmbeddings(), collection_name="test"
# )

# data = newDB.get()

# print(data)
