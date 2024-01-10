# Ejemplo de uso de un prompt que se conecta a Confluence y genera Historias de usuario y criterios de aceptación del requerimiento que se le mencione

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

from langchain.vectorstores import Chroma
from colorama import Fore

import pathlib
import sys

_parentdir = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(_parentdir))
print(_parentdir)

from scripts.config import Config

cfg = Config()

loader = ConfluenceLoader(
    url=cfg.jira_site, username=cfg.jira_user, api_key=cfg.jira_api_key
)

# Setear la key del espacio de trabajo de confluence en space_key
# limit es la cantidad de documentos a cargar consulta que hará loader, no el total de documentos a traer.
docs = loader.load(
    space_key="VIF",
    include_attachments=False,
    limit=50,
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000, 
    chunk_overlap=0, 
    separators=["", " ", "\n","\n\n", "(?<=\. )"]
)

texts = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embeddings)

retriever = db.as_retriever()

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
template = """Actua como un ingeniero de Software profesional que esta realizando una documentación de nuevos requerimientos. 
Tono: profesional
Tareas:
- Crear al menos 10 historias de usuario de la funcionalidad de esco fondos {query}
- Listar los principales criterios de aceptación
- Tener en cuenta los criterios de aceptación de la historia de usuario
- Realizar un resumen de cada historia de usuario para comprender su alcance


Título de la historia de usuario: <título-historia-usuario>

Resumen: <resumen-historia-usuario>
Criterios de Aceptación: <criterios-de-aceptacion>
"""

# Tipo de elemento de trabajo: Caso de Prueba
# Título: <descripcion>

# Pasos de prueba:
# Acción del paso: <Pasos-de-reproducción>
# Paso esperado: <Resultado-esperado>

prompt_template = PromptTemplate(
    template=template, input_variables=["query"], validate_template=False
)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

while True:
    print(Fore.WHITE)
    query = input("> ")
    answer = qa.run(prompt_template.format(query=query))

    print(Fore.GREEN, answer)
