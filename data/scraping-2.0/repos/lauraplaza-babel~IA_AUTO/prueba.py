from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
import config
from langchain.document_loaders import GitLoader
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from langchain.document_loaders import WebBaseLoader
from git import Repo
from msrest.authentication import BasicAuthentication
from azure.devops.connection import Connection
from msrestazure.azure_active_directory import AADTokenCredentials
from azure.devops.connection import Connection
from azure.devops.exceptions import AzureDevOpsServiceError
import os

 

 

 

api = config.OPENAI_API_KEY

 

 

def copy_file_to_txt(input_file, output_file):
    with open(input_file, 'r') as file:
        content = file.read()

 

    with open(output_file, 'w') as output:
        output.write(content)

 

    print(f"El archivo {input_file} se ha copiado en {output_file} correctamente.")

 

def obtener_contenido_archivo(nombre_archivo):
    with open(nombre_archivo, 'r') as archivo:
        contenido = archivo.read()
    return contenido

 

 

#TENGO EL REPO
loader = GitLoader(
    clone_url="https://dev.azure.com/Tailspin0523388/_git/terraform",
    repo_path="./REPOSITORIO/",
    branch="master",
     #file_filter=lambda file_path: file_path.endswith("Product.cs"),
)

 

data = loader.load()

 

 

 

#PASO A TXT EL ARCHIVO QUE QUIERO

 

input_file = 'REPOSITORIO/src/PartsUnlimited.Models/Product.cs'  # Reemplaza 'tu_archivo.txt' por la ruta del archivo que deseas copiar.
output_file = 'codigo.txt'
copy_file_to_txt(input_file, output_file)

 

 

 

# GUARDO EN UN STRING EL TXT
nombre_archivo = 'codigo.txt'  # Reemplaza 'archivo.txt' con el nombre del archivo que deseas leer.
codigo = obtener_contenido_archivo(nombre_archivo)
print(codigo)

 

 

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="Eres una IA hablando con un programador, la respuesta dada debe ser únicamente código modificado, no  pongas nada de texto ni comentarios"), # The persistent system prompt
    MessagesPlaceholder(variable_name="chat_history"), # Where the memory will be stored.
    HumanMessagePromptTemplate.from_template("{human_input}"), # Where the human input will injectd
])

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

 

llm = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=.7,openai_api_key=api)

 

chat_llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)

 

#print(chat_llm_chain.run(human_input=tarea + "Este es el código:" + "\n---\n" + codigo + "\n---\n"))

 

 

 

contador = 0
correcto = False

 

tarea= "Añademe varios comentario indicandome que crees que significa cada variable del código.:"
#while contador <=3 and correcto == False :
respuesta= chat_llm_chain.run(human_input=tarea + "Este es el código:" + "\n---\n" + codigo + "\n---\n")
print(respuesta)
    ## hacer un commit del pipeline
    ## por tanto se lanza el pipeline
    ##  tener log del pipeline
    ## if ha pasado todas las pruebas:
        # correcto= true
    #else:  
        # if error compilacion :
            # tarea= "El código que me has dado : " + respuesta + " tiene errores de compilación. solucionalo"
        #else error pruebas:
            # tarea= "El código que me has dado : " + respuesta + " no ha pasado las pruebas"

 

        #contador = contador +1