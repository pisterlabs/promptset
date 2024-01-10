import os
from sqlalchemy import create_engine
from langchain.chat_models.openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain


# Configurar la conexión a la base de datos
db_connection_string = "mssql+pyodbc://sa:220631@localhost/Finanzas?driver=SQL Server Native Client 11.0"
db_uri = "mssql+pyodbc://sa:220631@localhost/Finanzas?driver=SQL Server Native Client 11.0"

#Configurar la API DE CHATOPENAI
os.environ["OPENAI_API_KEY"] = "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"
model_name = "gpt-3.5-turbo"

#Probar conexion a la base de datos

try:
    engine = create_engine(db_connection_string)
    connection = engine.connect()
    print("Conexión exitosa a la base de datos.")
    connection.close()
except Exception as e:
    print(f"Error al conectar a la base de datos: {e}")


#Crear el modelo chatopenai
openai = ChatOpenAI(model_name=model_name)

#Crear la cadena de SQLDataBaseChain
memory = ConversationBufferWindowMemory(k=5)
db = SQLDatabase.from_uri(db_uri)
db_chain = SQLDatabaseChain.from_llm(openai, db, memory=memory, verbose=False, top_k=5)

# Formato personalizado de respuestas
# Formato personalizado de respuestas
def formato_consulta(question):
    return f"""
    **Siempre**
    Dada una pregunta del usuario:
    1. Crea una consulta de SQL Server.
    2. Revisa los resultados.
    3. Devuelve el dato.
    4. Si es necesario, proporciona aclaraciones o cualquier texto en español.

    Pregunta del usuario: "{question}"
    """

# Función para hacer la consulta
def consulta(input_usuario):
    mensaje_formateado = formato_consulta(input_usuario)
    resultado = db_chain.run(mensaje_formateado)
    return resultado