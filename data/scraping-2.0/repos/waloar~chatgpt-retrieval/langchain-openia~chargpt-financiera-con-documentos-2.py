import os
import sys

import openai
import json
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader,DirectoryLoader,CSVLoader

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

import constants
# REF https://blog.devgenius.io/chat-with-document-s-using-openai-chatgpt-api-and-text-embedding-6a0ce3dc8bc8
# ref https://levelup.gitconnected.com/langchain-for-multiple-pdf-files-87c966e0c032
# REF https://towardsdatascience.com/4-ways-of-question-answering-in-langchain-188c6707cc5a

os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

query = None
if len(sys.argv) > 1:
    query = sys.argv[1]

#text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator=".")

loader = DirectoryLoader('./data/', glob="**/*.txt", loader_cls=TextLoader)

csvloader = CSVLoader(file_path='./data/pp/nombre-de-comercios.csv',csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": ["nombre de comercio","Calle","Numero","localidad","rubro"],
    },)

raw_documents = loader.load()
print(len(raw_documents))

csv_documents = csvloader.load()
print(len(csv_documents))

raw_documents.extend(csv_documents)

# Divide los documentos en trozos de 1000 tokens
documents=text_splitter.split_documents(raw_documents)

# documents=raw_documents

embeddings = OpenAIEmbeddings()
# Vectoriza los documentos
vectordb = Chroma.from_documents(
    documents, embeddings)

# vectordb = Chroma.from_documents(
#     documents, embeddings, persist_directory='persist')
# vectordb.persist()

# Aplica el modelo de ChatGPT para conversacion
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)


# Construye un templeta de prompt
template = """ Debes actuar como un agente de atencion a clientes. Debes dirigirte al interlocutor de una manera cordial y atenta, en lo posible por su nombre.
Utiliza todas las piezas del texto y de la Historia para responder las preguntas. Si no contestas la pregunta, simplemente di que no lo sabes y que puedes consultarlo con un supervisor,no trates de crear una respuesta. Utiliza tres oraciones como maximo.
 Manten las respuestas concisas.
 Debes tener en consideracion:
 1. Si el interlocutor quiere sacar un prestamo, o desea pagar una deuda, o saber el estado de su cuenta, verifica si tienes el dni y nombre del texto y de la historia, si no lo tienes, debes pedirle su nombre completo y su DNI.
 2. Si el interlocutor desea que le transfieras con un agente de ventas, o un supervisor, asegurate de tener el <dni> <nombre> y <apellido>, debes pedirle amablemente su nombre completo y su DNI y luego finalizar la conversacion.
 Ademas debes:
 1. Resumir el texto en una oracion como maximo y etiquetalo como <resumen>.
 2. Si existen nombres de personas separa el nombre de pila y etiquetarlo como <nombre> y el o los apellidos como <apellido>.
 3. Si existe parte del texto numeros de 7 u 8 caracteres etiquetarlo como <dni> y formatearlo como xx.xxx.xxx .
 4. Extraer la intencion en un maximo de 2 palabras y etiquetalo como <intencion>.
 6. Analiza el senitmiento de la pregunta y etiquetalo como <sentimiento>.
 5. En todos los casos, genera una respuesta amigable con una frase amigable en una etiqueta <frase>.
 6. Si cononces el nombre de la persona incluye su nombre en la respuesta <frase>.
 7. La <frase> no puede ser mas de 2 oraciones.
 9. Devoler un JSON con los siguientes campos resumen: <resumen> intencion: <intencion> sentimiento: <sentimiento> nombre: <nombre> apellido: <apellido> dni: <dni> respuesta: <frase>. En el caso que no se pueda extraer algun valor del texto y su historia, se debe devolver el campo con un string vacio.
 
 Context: {context}
 History: {chat_history}
 
 Question: {question}
 
"""


QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=template,

)
# QA_CHAIN_PROMPT = PromptTemplate.from_template(template)


memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, input_key="question")


# Utiliza la extraccion de datos conversacionales de pregunta y respuesta.
chain = RetrievalQA.from_chain_type(
    llm,
    chain_type='stuff',
    retriever=vectordb.as_retriever(search_kwargs={"k": 1}),
    return_source_documents=True,
    verbose=False,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT, "memory": memory, "verbose": True}
    )
   

chat_history = []
while True:
    if not query:
        query = input("Prompt: ")
    if query in ['quit', 'q', 'exit']:
        sys.exit()
    result = chain({"query": query, "chat_history": chat_history})
    # result = chain({"query": query})
    # print(result['result'])
    resultado =result['result']
    if "respuesta" in resultado:
        dato= json.loads(resultado)
        print(dato["respuesta"])
    else:
        print(result['result'])

    # print(result['result']['respuesta'])    
    

    # print(chain.combine_documents_chain.memory)

    # chat_history.append((query, result['result']))
    query = None
