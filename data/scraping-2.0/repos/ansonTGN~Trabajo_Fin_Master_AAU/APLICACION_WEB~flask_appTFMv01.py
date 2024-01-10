# Bibliotecas a Cargar

# Group 1: Web Framework (Flask) / Grupo 1: Framework Web (Flask)
from flask import Flask, request, jsonify, render_template

# Group 2: System Operations and Environment Variables / Grupo 2: Operaciones del Sistema y Variables de Entorno
import os
from dotenv import load_dotenv, find_dotenv
import json

# Group 3: Natural Language Processing (NLP) / Grupo 3: Procesamiento de Lenguaje Natural (NLP)
from langchain import (
    PromptTemplate, LLMChain, llms, text_splitter, vectorstores, chains, embeddings
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Group 4: Data Loading and Database Operations / Grupo 4: Carga de Datos y Operaciones de Base de Datos
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.graphs import Neo4jGraph
from neo4j import GraphDatabase

# Group 5: Additional Language Processing / Grupo 5: Procesamiento de Lenguaje Adicional
from langchain.llms import CTransformers
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
# Nota importante: 
#      - HuggingFaceEmbeddings utiliza el paquete sentence_transformers para generar incrustaciones localmente,
#      - HuggingFaceHubEmbeddings utiliza la API de inferencia de HuggingFace (Problemas con OpenAI API!!!)
#from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.embeddings import SelfHostedHuggingFaceInstructEmbeddings
#import runhouse as rh
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.prompt_selector import ConditionalPromptSelector


# ----------------------------------------------------
# Carga de variables de Entorno 
# Load environment variables

load_dotenv(find_dotenv(), override=True)

# Accede a las variables de entorno en tu código
# Access environment variables in your code
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
URL_NEO4J = os.getenv('URL_NEO4J')
USERNAME_NEO4J = os.getenv('USERNAME_NEO4J')
PASSWORD_NEO4J = os.getenv('PASSWORD_NEO4J')

# ----------------------------------------------------
# Initialize LLM and other components as in the original code
# Inicializa el LLM y otros componentes como en el código original

# local_llm = "neural-chat-7b-v3-1.Q4_K_M.gguf"
local_llm = "openhermes-2.5-mistral-7b.Q3_K_M.gguf"

# Define los parámetros de configuración para el modelo
config = {
    'max_new_tokens': 512,       # Maximum number of new tokens generated
    'repetition_penalty': 1.1,   # Repetition penalty for text generation
    'temperature': 0,           # Temperature for controlling randomness (0 for deterministic output)
    'context_length': 1024,     # Length of the input context
    'stream': False             # Whether to stream the output or generate it all at once
}

# Initialize a CTransformers object with the specified parameters.
# Inicializa un objeto CTransformers con los parámetros especificados.
llm = CTransformers(
    model=local_llm,
    model_type="mistral",
    lib="avx2",
    **config
)

# Print a message to indicate that LLM (Language Model) has been initialized.
# Imprime un mensaje para indicar que LLM (Modelo de Lenguaje) ha sido inicializado.
print("LLM Initialized....")

# ----------------------------------------------------

# Define the model name for embeddings.
# Define el nombre del modelo para los embeddings.
#model_name = "BAAI/bge-large-en"
model_name = "BAAI/bge-large-en-v1.5"

# Specify keyword arguments for the model (e.g., device).
# Especifica argumentos clave para el modelo (por ejemplo, dispositivo).
model_kwargs = {'device': 'cpu'}

# Specify keyword arguments for encoding (e.g., normalization settings).
# Especifica argumentos clave para la codificación (por ejemplo, configuración de normalización).
encode_kwargs = {'normalize_embeddings': True}

#model_name = "hkunlp/instructor-large"
#gpu = rh.cluster(name='rh-a10x', instance_type='A100:1')
#hf = SelfHostedHuggingFaceInstructEmbeddings(model_name=model_name, hardware=gpu)
#embedding = hf.embed_query("hi this is harrison")

# Initialize HuggingFaceBgeEmbeddings with the specified parameters.
# Inicializa HuggingFaceBgeEmbeddings con los parámetros especificados.
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# ----------------------------------------------------

# Define a prompt template for generating prompts.
# Define una plantilla de prompt para generar preguntas.
prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Specify the input variables used in the template.
# Especifica las variables de entrada utilizadas en la plantilla.
input_variables = ['context', 'question']

# Create a PromptTemplate object with the specified template and input variables.
# Crea un objeto PromptTemplate con la plantilla y las variables de entrada especificadas.
prompt = PromptTemplate(template=prompt_template, input_variables=input_variables)

# ----------------------------------------------------

# Establish a connection to the Neo4j database.
# Establece una conexión a la base de datos Neo4j.
url = URL_NEO4J
username = USERNAME_NEO4J
password = PASSWORD_NEO4J

# Function to check the connection to the Neo4j database
# Función para verificar la conexión a la base de datos Neo4j
def verificar_conexion(url, username, password):
    try:
        # Attempt to establish a connection to the Neo4j database
        # Intento de establecer una conexión a la base de datos Neo4j
        with GraphDatabase.driver(url, auth=(username, password)) as driver:
            with driver.session() as session:
                # Run a test query to check the connection
                # Ejecutar una consulta de prueba para verificar la conexión
                session.run("RETURN 1")
        return True  # Return True if the connection is successful
    except Exception as e:
        # Handle any exceptions that occur during the connection attempt
        # Manejar cualquier excepción que ocurra durante el intento de conexión
        print(f"Connection error: {str(e)}")
        return False  # Return False if the connection fails

# Check the connection to the Neo4j database
# Verifica la conexión a la base de datos Neo4j
if verificar_conexion(url, username, password):
    print("Successful connection to the Neo4j database")
else:
    print("Failed to connect to the Neo4j database")

# ----------------------------------------------------

# Function to initialize the embedding for a node type
# Función para inicializar el embedding para un tipo de nodo
def init_node_embedding(node_label, text_properties):
    return Neo4jVector.from_existing_graph(
        embedding=embeddings,  # The embedding to use for initialization
        url=url,  # URL for the Neo4j database
        username=username,  # Username for database authentication
        password=password,  # Password for database authentication
        index_name=f"{node_label.lower()}_index",  # Name of the index for the node label
        node_label=node_label,  # Label of the node type
        text_node_properties=text_properties,  # Text properties for node embedding
        embedding_node_property="NEWembedding"  # Property name for the embedding in the database
    )

# Initialize embeddings for different types of nodes
# Inicializar embeddings para diferentes tipos de nodos

# Initialize embedding for "Choice" nodes with support for Spanish strings
Choice_embedding = init_node_embedding("Choice", ["string_es"])
print("Choice embedding OK")  # Print message to indicate successful initialization
# Inicializar la incrustación para nodos "Choice" con soporte para cadenas en español
# Imprimir mensaje para indicar una inicialización exitosa

# Initialize embedding for "Question" nodes with support for Spanish strings
Question_embedding = init_node_embedding("Question", ["string_es"])
print("Question embedding OK")  # Print message to indicate successful initialization
# Inicializar la incrustación para nodos "Question" con soporte para cadenas en español
# Imprimir mensaje para indicar una inicialización exitosa

# Initialize embedding for "Theme" nodes with support for Spanish strings
Theme_embedding = init_node_embedding("Theme", ["string_es"])
print("Theme embedding OK")  # Print message to indicate successful initialization
# Inicializar la incrustación para nodos "Theme" con soporte para cadenas en español
# Imprimir mensaje para indicar una inicialización exitosa

# Initialize embedding for "Reply" nodes with support for Spanish strings
Reply_embedding = init_node_embedding("Reply", ["string_es"])
print("Reply embedding OK")  # Print message to indicate successful initialization
# Inicializar la incrustación para nodos "Reply" con soporte para cadenas en español
# Imprimir mensaje para indicar una inicialización exitosa

# ----------------------------------------------------

# Function to calculate similarity between the query and nodes
# Función para calcular la similitud entre la consulta y los nodos
def calculate_similarity(tx, input_query):
    # Get embeddings for the query
    # Obtener representaciones incrustadas (embeddings) para la consulta
    encode_kwargs = {'normalize_embeddings': False}
    query_embedding = embeddings.embed_query(input_query)
    #query_embedding = hf.embed_query(input_query)

    # Obtain information for all nodes
    # Obtener información de todos los nodos
    result = tx.run("""
    MATCH (n)
    RETURN ID(n) AS node_id, n.NEWembedding AS node_embedding
    """)

    queries_to_update = []

    for record in result:
        node_id = record["node_id"]
        node_embedding = record["node_embedding"]

        # Calculate cosine similarity between the query and the node
        # Calcular la similitud del coseno entre la consulta y el nodo
        similarity_score = cosine_similarity([query_embedding], [node_embedding])[0][0]

        # Prepare the query to update the node
        # Preparar la consulta para actualizar el nodo
        queries_to_update.append({
            "node_id": node_id,
            "score": similarity_score
        })

    # Update all nodes in a single transaction
    # Actualizar todos los nodos en una sola transacción
    for query_data in queries_to_update:
        tx.run("""
        MATCH (n)
        WHERE ID(n) = $node_id
        SET n.similarity_query = $score
        """, node_id=query_data["node_id"], score=query_data["score"])

# ----------------------------------------------------

# Function to calculate the list of closest nodes
# Función para calcular la lista de nodos más cercanos
def calculate_top_similarity_nodes(tx, input_query):
    # Get embeddings for the query
    query_embedding = embeddings.encode(input_query, **encode_kwargs)

    result = tx.run("""
    MATCH (n)
    RETURN n.node_id AS node_id, n.NEWembedding AS node_embedding, n.similaridadquery AS similarity_score, n.string_es AS node_text
    ORDER BY similarity_score DESC
    LIMIT 5
    """)

    top_nodes = []

    for record in result:
        node_id = record["node_id"]
        node_embedding = record["node_embedding"]
        similarity_score = record["similarity_score"]
        textoNodo = record["node_text"]

        top_nodes.append({
            "node_id": node_id,
            "node_embedding": node_embedding,
            "similarity_score": similarity_score,
            "texto": textoNodo
        })

    return top_nodes

# ----------------------------------------------------

class GraphPathFinder:
    def __init__(self, url, user, password):
        # Initialize the GraphPathFinder class with database connection parameters
        # Inicializar la clase GraphPathFinder con parámetros de conexión a la base de datos
        self.driver = GraphDatabase.driver(url, auth=(user, password))

    def close(self):
        # Close the database connection
        # Cerrar la conexión a la base de datos
        self.driver.close()

    def find_paths(self, start_node_id, max_depth, external_embedding):
        # Nota solo localiza nodos con similaridadquery >= 0.01
        with self.driver.session() as session:
            query = """
            MATCH path = (n {node_id: $start_node_id})-[*1..%s]-(end)
            WHERE all(node in nodes(path) WHERE node.similarity_query >= 0.01)
            RETURN [node in nodes(path) | [node.node_id, node.similarity_query, node.string_es]] AS node_info
            ORDER BY reduce(s = 0, n IN nodes(path) | s + n.similarity_query) DESC
            """ % max_depth
            result = session.run(query, start_node_id=start_node_id)
            # Return a list of node information for the found paths
            # Devolver una lista de información de nodos para los caminos encontrados
            return [record["node_info"] for record in result]

    def get_all_theme_strings(self):
        with self.driver.session() as session:
            query = """
            MATCH (theme:Theme)
            RETURN collect(theme.string_es) AS theme_strings
            """
            result = session.run(query)
            # Return a list of all theme strings in the database
            # Devolver una lista de todas las cadenas de temas en la base de datos
            return [record["theme_strings"] for record in result][0]

    def get_node_ids_by_theme_string(self, theme_string):
        with self.driver.session() as session:
            query = """
            MATCH (node:Theme {string_es: $theme_string})
            RETURN collect(node.node_id) AS node_ids
            """
            result = session.run(query, theme_string=theme_string)
            # Return a list of node IDs associated with the given theme string
            # Devolver una lista de IDs de nodos asociados con la cadena de tema proporcionada
            return [record["node_ids"] for record in result][0]
        
# ----------------------------------------------------

def procesar_nodos1(finder, start_node_ids, Profund_Maxima_Nodos, n_top_paths, query, clasificacion, external_embedding=None):
    """
    finder --> Clase de analisis caminos
    start_nodes_ids --> Nodo inicial donde empezar la busqueda
    Profund_Maxima_Nodos --> máximo nivel de profundidad de los nodos
    n_top_paths --> Numero de caminos a devolver
    query --> Consulta del usuario
    external_embedding --> A utilizar en el futuro
    """

    resultados = []  # Almacenará los resultados finales

    # Itera sobre los nodos iniciales
    for start_node_id in start_node_ids:
        # Encuentra los caminos desde el nodo inicial
        paths = finder.find_paths(start_node_id, Profund_Maxima_Nodos, external_embedding)

        # Calcula la media de similaridad y almacena junto con el camino
        path_with_mean = []
        for path in paths:
            if path:
                # Calcula la media de la similaridad en el camino
                mean_similarity = sum(node[1] for node in path if node[1] is not None) / len(path)
                path_with_mean.append((path, mean_similarity))

        # Ordena los caminos por la media de similaridad de mayor a menor
        path_with_mean.sort(key=lambda x: x[1], reverse=True)

        # Guarda la información de los n caminos con mayor media
        top_paths = path_with_mean[:n_top_paths]
        for path, mean in top_paths:
            path_info = [(node[0], node[1]) for node in path]
            string_es_values = [node[2] for node in path if node[2] is not None]
            max_similarity_node = max(path, key=lambda x: x[1])
            max_similarity_node_info = (max_similarity_node[0], max_similarity_node[1], max_similarity_node[2])

            resultados.append({
                'start_node_id': start_node_id,
                'path_info': path_info,
                'mean_similarity': mean,
                'string_es_values': string_es_values,
                'max_similarity_node_info': max_similarity_node_info,
                'query': query,
                'clasificacion': clasificacion
            })

    return resultados  # Devuelve los resultados finales

# ----------------------------------------------------

# Carga Modelo Open AI

#modelOpenAI = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

modelOpenAI = ChatOpenAI(temperature=0)

# ----------------------------------------------------

DEFAULT_SEARCH_PROMPT = PromptTemplate(
    input_variables=["texto", "categorias"],
    template=""""Clasifica el siguiente texto en una de las siguientes categorías {categorias}: \n Texto: {texto}\nCategoría: """,
)

chainOpenAI = DEFAULT_SEARCH_PROMPT | modelOpenAI | StrOutputParser()

# ----------------------------------------------------

def clasificar_LLMs(texto, categorias, chain):
    '''Función de clasificación
    Entrada:
        texto -> Conversación Usuaria
        categorias --> Lista de categorias donde se ha de clasificar
        chain --> Cadena de LangChain
    respuesta:
        Respuesta del modelo
    '''
    response = chain.invoke({"categorias": categorias, "texto": texto})
    return response

# ----------------------------------------------------

# Creación de la instancia de GraphPathFinder
finder = GraphPathFinder(url, username, password)
categorias = finder.get_all_theme_strings()
categorias = list(set(categorias))
#print(categorias)
print('Numero Categorias Distintas:', len(categorias))

# ----------------------------------------------------

# Funcion para generar la salida resultados

def imprimir_resultados_OK(indice_mensaje, mensaje, resultados, profundidad_maxima):
    print(f"RESULTADOS PARA EL MENSAJE {indice_mensaje} ('{mensaje}'):")
    print("Profundidad Máxima:", profundidad_maxima)
    print("--------------------------------------------------")
    for idx, resultado in enumerate(resultados, 1):
        print(f"Resultado {idx}: {resultado}")
    print("--------------------------------------------------\n")

def procesar_mensajes(mensajes, url, username, password, model_name, model_kwargs, categorias, chainOpenAI,
                      calculate_similarity, clasificar_LLMs, procesar_nodos1, mostrar_resultado):
    '''
    ENTRADA:
    mensajes                --> Lista de mensajes
    url                     --> URL de la base de datos Neo4j
    username                --> Nombre de usuario para la autenticación de la base de datos
    password                --> Contraseña para la autenticación de la base de datos
    model_name              --> Nombre del modelo para los embeddings
    model_kwargs            --> Argumentos clave para el modelo (por ejemplo, dispositivo)
    categorias              --> Lista de categorías para clasificar
    chainOpenAI             --> Cadena de LangChain
    calculate_similarity    --> Función para calcular la similaridad
    clasificar_LLMs         --> Función para clasificar los mensajes
    procesar_nodos1         --> Función para procesar los nodos
    mostrar_resultado       --> Función para mostrar los resultados

    SALIDA:
    resultados              --> Lista de resultados
    '''

    # Inicializar lista para almacenar los resultados de cada mensaje
    resultados_totales = []

    # Para cada mensaje en la lista de mensajes
    for mensaje in mensajes:
        # Imprimir el mensaje
        print("Procesando MENSAJE:", mensaje)
        print("-------------------------")

        # Query que deseas comparar con las propiedades de los nodos
        query = mensaje

        # Conexión a la base de datos Neo4j y cálculo de la similitud del coseno para el query
        with GraphDatabase.driver(url, auth=(username, password)) as driver:
            embeddings = SentenceTransformer(model_name, **model_kwargs)

            with driver.session() as session:
                session.execute_write(calculate_similarity, query)

        print(f"Similitud del coseno calculada y asignada en la base de datos Neo4j para el query: {query}")

        # Identificación de los temas de la pregunta
        salidaOpen = clasificar_LLMs(mensaje, categorias, chainOpenAI)
        print('Tematica:', salidaOpen)

        # Creación de la instancia de GraphPathFinder
        finder = GraphPathFinder(url, username, password)
        ids = finder.get_node_ids_by_theme_string(salidaOpen)
        print('Lista de Nodos:', ids)

        # Lista de identificadores de nodo iniciales
        start_node_ids = ids

        # Profundidad máxima de búsqueda
        Profund_Maxima_Nodos = 7
        print('Profundidad Máxima de Nodos:', Profund_Maxima_Nodos)

        # Número de caminos con mayor media de similaridad a mostrar
        n_top_paths = 5
        print('Número de caminos con mayor media de similaridad a mostrar:', n_top_paths)

        # Llamada a la función procesar_nodos
        resultados = procesar_nodos1(finder, start_node_ids, Profund_Maxima_Nodos, n_top_paths, query, salidaOpen)

        # Agregar el mensaje y sus resultados a la lista de resultados totales
        resultados_mensaje = {
            'mensaje': mensaje,
            'resultados': resultados
        }
        resultados_totales.append(resultados_mensaje)

        # Cerrar la conexión
        finder.close()

    # Retornar la lista de resultados totales
    return resultados_totales

# ----------------------------------------------------

def generar_salida_json3(resultados, n_top_paths=3):
    """
    Genera una salida JSON a partir de una nueva estructura de lista de resultados y un mensaje asociado.

    :param resultados: Lista de diccionarios con los resultados y mensajes a procesar.
    :param n_top_paths: Número de elementos máximos a considerar en la salida.
    :return: Cadena en formato JSON con los datos procesados.
    """
    output_data = {'Mensajes_Procesados': []}

    for resultado in resultados:
        mensaje = resultado['mensaje']
        datos_resultado = resultado['resultados']

        mensaje_data = {
            'Contenido': mensaje,
            'Resultados_Analizados': []
        }

        for res in datos_resultado:
            # Asegurarse de que no se exceda el número de resultados disponibles
            n_top_paths = min(n_top_paths, len(res['path_info']))

            resultados_analizados = {
                'Indice': res['start_node_id'],
                'Clasificacion': res['clasificacion'],
                'Cadenas_Analizadas': [],
                'Nodo_Mayor_Similitud': {
                    'Indice': res['max_similarity_node_info'][0],
                    'Similaridad': res['max_similarity_node_info'][1],
                    'Texto': res['max_similarity_node_info'][2]
                },
                'Similaridad_Media': res['mean_similarity']
            }

            for n in range(n_top_paths):
                nodo_id, similaridad = res['path_info'][n]
                texto = res['string_es_values'][n]
                nodo_data = {
                    'Indice_Nodo': nodo_id,
                    'Similaridad': similaridad,
                    'Texto': texto
                }
                resultados_analizados['Cadenas_Analizadas'].append(nodo_data)

            mensaje_data['Resultados_Analizados'].append(resultados_analizados)

        output_data['Mensajes_Procesados'].append(mensaje_data)

    # Serializar los datos de salida como JSON
    json_output = json.dumps(output_data, ensure_ascii=False, indent=4)
    return json_output
# ----------------------------------------------------

def mostrar_resultado(resultado, mostrar):
    ''' Imprime el resultado de la búsqueda de caminos en la base de datos Neo4j
    limitando el número de nodos del camino mostrados
    y valores de string_es a mostrar.
    el valor limite dado por: 'mostrar'
    '''
    print("ID del Nodo Inicial:", resultado['start_node_id'])

    num_nodos = min(len(resultado['path_info']), mostrar)
    print("\nInformación del Camino (mostrando hasta {} nodos):".format(num_nodos))
    for node in resultado['path_info'][:num_nodos]:
        print(f"  Nodo ID: {node[0]}, Similaridad: {node[1]}")

    print("\nMedia de Similaridad:", resultado['mean_similarity'])

    num_valores_string = min(len(resultado['string_es_values']), mostrar)
    print("\nValores de string_es (mostrando hasta {} valores):".format(num_valores_string))
    for value in resultado['string_es_values'][:num_valores_string]:
        print(f"  {value}")

    print("\nInformación del Nodo con Máxima Similaridad:")
    max_node_info = resultado['max_similarity_node_info']
    print(f"  Nodo ID: {max_node_info[0]}, Similaridad: {max_node_info[1]}")
    print(f"  string_es: {max_node_info[2]}")
    return

# ----------------------------------------------------

from langchain.llms import OpenAI

# Inicializa el modelo de OpenAI
openai_model = OpenAI(temperature=0, model="text-davinci-003")

def resumir_texto(texto):
    prompt = f"Eres un experto en lactancia. Resume el texto: {texto}\nIndica las ideas clave. No respondas a las preguntas."
    
    # Utiliza el modelo para generar una respuesta
    respuesta = openai_model(prompt, max_tokens=250)
    
    # Extrae y retorna el texto de la respuesta
    return respuesta

# ----------------------------------------------------

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

vector_index = Neo4jVector.from_existing_graph(
    embeddings,
    url=url,
    username=username,
    password=password,
    index_name='Indices',
    node_label=["Choice", "Question", "Reply", "Theme"],
    text_node_properties=['string_es'],
    embedding_node_property='NEWembedding',
)

vector_qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(), chain_type="stuff", retriever=vector_index.as_retriever())

# ----------------------------------------------------

# Función para procesar el JSON y extraer la cadena con mayor similaridad media por contenido
def procesar_jsonRECORTAR(input_json):
    mensajes_procesados = input_json.get("Mensajes_Procesados", [])
    
    resultado_final = []

    for mensaje in mensajes_procesados:
        contenido = mensaje.get("Contenido")
        resultados_analizados = mensaje.get("Resultados_Analizados", [])
        
        # Inicializar variables para encontrar la cadena con mayor similaridad media
        cadena_mayor_similaridad = None
        similaridad_maxima = -1

        for resultado in resultados_analizados:
            similaridad_media = resultado.get("Similaridad_Media", 0)
            if similaridad_media > similaridad_maxima:
                similaridad_maxima = similaridad_media
                cadena_mayor_similaridad = resultado

        # Agregar al resultado final solo si se encontró una cadena
        if cadena_mayor_similaridad:
            resultado_final.append({
                "Contenido": contenido,
                "Resultados_Analizados": [cadena_mayor_similaridad]
            })

    return {"Mensajes_Procesados": resultado_final}



# ----------------------------------------------------

app = Flask(__name__)

# Variable global
valor_global = False

@app.route('/')
def index():
    return render_template('TFM_02.html')

# Ruta para asignación de valor
@app.route('/set_valor', methods=['POST'])
def set_valor():
    global valor_global
    data = request.get_json()
    valor_global = data['valor']
    return jsonify({"message": "Valor actualizado correctamente", "nuevo_valor": valor_global})


@app.route('/post_message', methods=['POST'])
def post_message():
    try:
        # Extrayendo el mensaje del cuerpo de la solicitud POST
        data = request.json
        mensaje = data.get('mensaje', '')
        print("Mensaje recibido:", mensaje, '\n')  # Para depuración

        mensajes = [mensaje]

        # Frente a mensajes complejos es mejor resumir antes de procesar
        mensaje_preprocesado = []

        for mensaje in mensajes:
            respuesta = resumir_texto(mensaje)
            print('Mensaje:',mensaje)
            print('Respuesta:', respuesta)
            mensaje_preprocesado.append(respuesta)

        # Dividir el texto por retornos de línea y eliminar espacios al principio y final de cada línea
        mensaje_1 = [linea.strip() for elemento in mensaje_preprocesado for linea in elemento.split('\n') if linea.strip()]
        print("Mensaje recibido dividido:", mensaje_1, '\n')  # Para depuración

        resultados_todos = []
        mensajeLISTA = []
        for mensajelista in mensaje_1:
            print("Mensaje recibido resumido:", mensajelista, '\n')  # Para depuración
            mensajeLISTA.append(mensajelista)

        #mensajes = [mensajeLISTA[0]]
        #print("Mensaje recibido a procesar:", mensajes, '\n')  # Para depuración

        ## Funciones de procesamiento (asegúrate de que funcionen correctamente)
        #resultados = procesar_mensajes(mensajes, url, username, password, model_name, model_kwargs, categorias, chainOpenAI,
        #              calculate_similarity, clasificar_LLMs, procesar_nodos1, mostrar_resultado)
        #print("Resultados_OK")  # Para depuración
        ## Generar salida JSON
        #json_output = generar_salida_json3(resultados, n_top_paths=3)
        #print("Salida JSON generada")  # Para depuración
        #print(json_output)
        #return jsonify(json_output)
            
        # Procesar todos los mensajes de la lista
        #resultados_todos = []
        #for mensaje in mensajeLista:
        #print("Mensaje recibido a procesar:", mensaje, '\n')  # Para depuración
        resultados = procesar_mensajes(mensajeLISTA, url, username, password, model_name, model_kwargs, categorias, chainOpenAI,
                                   calculate_similarity, clasificar_LLMs, procesar_nodos1, mostrar_resultado)
        #    resultados_todos.extend(resultados)
        print("Resultados_OK")  # Para depuración
        # Generar salida JSON para todos los resultados
        json_output = generar_salida_json3(resultados, n_top_paths=3)

        # Filtrar salida JSON para obtener la cadena con mayor similaridad media por contenido
        #resultado_json = procesar_jsonRECORTAR(json_output)
        #print(json.dumps(resultado_json, indent=4))
        
        print("Salida JSON generada")  # Para depuración
        print(json_output)
        return jsonify(json_output)


    except Exception as e:
        print("Error en post_message:", e)
        return jsonify({"error": str(e)})

@app.route('/post_resumen', methods=['POST'])
def post_resumen():
    try:
        # Extrayendo el mensaje del cuerpo de la solicitud POST
        data = request.json
        mensaje = data.get('mensaje', '')
        print("Mensaje recibido:", mensaje)  # Para depuración

        mensajes = [mensaje]

        # Clases identificadas
        print('Clases identificadas:', categorias)
        # Mensaje
        print('Mensaje:', mensaje)

        mensaje_preprocesado = []

        for mensaje in mensajes:
            respuesta = resumir_texto(mensaje)
            print('Mensaje:',mensaje)
            print('Respuesta:', respuesta)
            mensaje_preprocesado.append(respuesta)

        # Envía la respuesta en un formato que la función 'displaySummaryResponse' espera
        json_resultado = {"resumenes": mensaje_preprocesado}
        print(json_resultado)
        return jsonify(json_resultado)
    except Exception as e:
        print("Error en post_resumen:", e)
        return jsonify({"error": str(e)})
    
@app.route('/post_chat', methods=['POST'])
def post_chat():
    try:
        # Extrayendo el mensaje del cuerpo de la solicitud POST
        data = request.json
        mensaje = data.get('mensaje', '')
        print("Mensaje recibido:", mensaje)  # Para depuración

        if valor_global:
            mensaje = 'Desde la perspectiva de una usuaria responde a: '+ mensaje
        else:
            mensaje = 'Indica preguntas para mejorar tu respuesta como experto en lactancia materna al mensaje: '+ mensaje

        mensajes = [mensaje]

        # Clasificación del mensaje
        salidaOpen= clasificar_LLMs(mensaje, categorias, chainOpenAI)
        # Identificación de los nodos
        idx = ''
        ids = finder.get_node_ids_by_theme_string(salidaOpen)
        for s in ids:
            print('Lista de Nodos:', s)
            idx = s
        ids_value = idx  # Asegúrate de que este valor sea seguro para evitar la inyección de SQL

        contextualize_query = f"""
            MATCH (n:Theme {{node_id: '{ids_value}'}})-[*3]-(m:Choice)
            WITH n.string_es AS self, REDUCE(s = "", item IN COLLECT(m.string_es) | s + "\n" + item) AS ctxt, m.score as score, {{}} as metadata limit 1
            RETURN self + ctxt AS text, score, metadata
        """

        #"This model's maximum context length is 4097 tokens. However, your messages resulted in 9852 tokens. Please reduce the length of the messages.

        contextualized_vectorstore = Neo4jVector.from_existing_index(
            embeddings,
            url=url,
            username=username,
            password=password,
            index_name='Indices',
            retrieval_query=contextualize_query,
        )

        vector_plus_context_qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(), chain_type="stuff", retriever=contextualized_vectorstore.as_retriever(k=1))
        
        response = vector_plus_context_qa.run(mensaje)

        #print('Contexto:', response)

        ## Crear una instancia de RetrievalQA
        #retrieval_qa = RetrievalQA()

        #print('Paso2')

        # Establecer el contexto
        #retrieval_qa.set_context([response])

        # Realizar una pregunta
        #pregunta = "Genera tres preguntas segun el contexto que te he pasado"
        #respuesta1 = retrieval_qa.ask_question(pregunta)

        # Imprimir la respuesta
        #print(respuesta1)

        respuesta = response
        #respuesta = respuesta + '|-|' +respuesta1
        print('Respuesta:', respuesta)

        # Funciones de procesamiento (asegúrate de que funcionen correctamente)
        # Procesar cada mensaje y almacenar el resultado
        mensaje_preprocesado = [respuesta]

        #for mensaje in mensajes:
        #    respuesta = vector_qa.run(mensaje)
        #    #response = vector_index.similarity_search(mensaje, k=5)
        #    #for s in response:
        #    #    print(s.page_content, s.metadata['node_id'])
        #    #    mensaje_preprocesado.append(s.metadata['node_id'])
        #    print('Mensaje:',mensaje)
        #    print('Respuesta:', respuesta)
        #    mensaje_preprocesado.append(respuesta)

        # Envía la respuesta en un formato que la función 'displaySummaryResponse' espera
        json_resultado = {"chat": mensaje_preprocesado}
        print(json_resultado)
        return jsonify(json_resultado)
    except Exception as e:
        print("Error en chat:", e)
        return jsonify({"error": str(e)})
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
