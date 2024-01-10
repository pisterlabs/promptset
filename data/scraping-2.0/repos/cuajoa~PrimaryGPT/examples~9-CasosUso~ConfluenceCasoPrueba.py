# Ejemplo de uso de un prompt que se conecta a Confluence y genera casos de prueba del requerimiento que se le asigne

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
    chunk_size=4000, chunk_overlap=0, 
    separators=["", " ", "\n","\n\n", "(?<=\. )"]
    
)

texts = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embeddings)

retriever = db.as_retriever()

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
template = """Actua como un ingeniero de QA profesional que esta realizando una documentación de nuevos requerimientos. 
Tono: profesional
Tareas:
- Crear al menos 10 casos de prueba de la funcionalidad de esco fondos {query}
- Indicar los pasos para reproducir el problema
- Los casos deben incluir una descripción del problema
- Los casos deben tener en cuenta las precondiciones
- Los casos deben tener en cuenta casos que no sean exitosos
- Los casos deben tener un resultado esperado

crear un archivo CSV con el siguiente formato:
Cabecera: Id.,Tipo de elemento de trabajo,Título,Paso de prueba,Acción del paso,Paso esperado
,Caso de prueba,<descripcion>,,,
,,,1,<Pasos-de-reproducción>,<Resultado-esperado>
,,,2,<Pasos-de-reproducción>,<Resultado-esperado>
,,,3,<Pasos-de-reproducción>,<Resultado-esperado>

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
