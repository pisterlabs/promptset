# 1. Cargar la bbdd con langchain
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chat_models import ChatOpenAI
from langchain.sql_database import SQLDatabase

# 2. Importar las APIs
import a_env_vars
import os
os.environ["OPENAI_API_KEY"] = a_env_vars.OPENAI_API_KEY

# 3. Crear el LLM
llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

# 4. Crear la cadena
db = SQLDatabase.from_uri("mysql+mysqlconnector://0yxorrwpqtf9fqzdzj9w:pscale_pw_heTDgUyo3eUVfabwSLIqGkbtIFj4RZUVDqoUaZgCmb9@aws.connect.psdb.cloud/errores_camiones")
cadena = SQLDatabaseChain.from_llm(llm=llm, db=db, verbose=False)

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
    # Verificar si la pregunta es frecuente
    if input_usuario.startswith("¿Cuál es el código de error") or input_usuario.startswith("¿Cómo soluciono el error"):
        # Es una pregunta frecuente, simplemente consulta el LLM
        resultado = cadena.run(input_usuario)
    else:
        # No es una pregunta frecuente, aplicar el formato personalizado
        consulta_personalizada = formato.format(question=input_usuario)
        resultado = cadena.run(consulta_personalizada)

    return resultado

# 7. Llamada a la función en un bucle infinito
if __name__ == "__main__":
    while True:
        pregunta_usuario = input("Ingrese una pregunta (o escriba 'salir' para salir): ")
        
        # Verificar si el usuario quiere salir
        if pregunta_usuario.lower() == 'salir':
            print("Saliendo del programa...")
            break  # Salir del bucle infinito
        
        resultado_consulta = consulta(pregunta_usuario)
        print("Resultado de la consulta:", resultado_consulta)
