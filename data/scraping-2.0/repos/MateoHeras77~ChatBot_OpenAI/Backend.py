# 1. Cargar la bbdd con langchain
from langchain.sql_database import SQLDatabase
db = SQLDatabase.from_uri("sqlite:///ecommerce.db")

# 2. Importar las APIs
import VarEntorno
import os
os.environ["OPENAI_API_KEY"] = VarEntorno.OPENAI_API_KEY

# 3. Crear el LLM
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature=0,model_name='gpt-3.5-turbo')

# 4. Crear la cadena
from langchain import SQLDatabaseChain
cadena = SQLDatabaseChain(llm = llm, database = db, verbose=False)

# 5. Formato personalizado de respuesta
formato = """
Data una pregunta del usuario:
1. crea una consulta de sqlite3
2. revisa los resultados
3. devuelve el dato
4. si tienes que hacer alguna aclaración o devolver cualquier texto que sea siempre en español
#{question}
"""

# 6. Función para hacer la consulta

def consulta(input_usuario):
    consulta = formato.format(question = input_usuario)
    resultado = cadena.run(consulta)
    return(resultado)