from dotenv import load_dotenv
import os
from os.path import abspath, dirname
import openai
import pandas as pd
from tqdm import tqdm
import json


# Cargo variables de configuración.
load_dotenv()

# Configuración.
os.environ["OPENAI_API_KEY"] = os.getenv('API_KEY')
openai.api_type = os.getenv('API_TYPE')
openai.api_version = os.getenv('API_VERSION')
openai.api_base = os.getenv('API_BASE')
openai.api_key = os.getenv('API_KEY')


def extraction(messages, engine=os.getenv('CHAT_ENGINE_16K'), temperature=0.1, top_p = 0.9):
    """
    Extracts information from a document containing test requirements and identifies all test cases and their expected results.

    Parameters:
    - messages (list): A list of messages exchanged between the user and the chatbot.
    - engine (str): The engine to use for generating responses. Default is CHAT_ENGINE_16K.
    - temperature (float): The temperature parameter for response generation. Default is 0.1.
    - top_p (float): The top-p parameter for response generation. Default is 0.9.

    Returns:
    - str: The extracted content containing the identified test cases and their expected results.
    """
    messages_full = [{"role": "system", "content": """Sos parte del equipo de testing de una compania de telecomunicaciones.
    - Vas a recibir un documento con los requerimientos para testeo de varios de los modulos de una aplicacion y debes identificar TODOS los casos de prueba presentes en él y su resultado esperado.
    """
    }] + messages

    timeout = 10

    try:
        response = openai.ChatCompletion.create(
            engine=engine,
            messages=messages_full,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout
        )

    except openai.OpenAIError as e:
        print("Error al realizar la solicitud:", str(e))
        return None
    
    # Medir la respuesta de los tokens.    
    prompt_tokens = response['usage']['prompt_tokens']
    completion_tokens = response['usage']['completion_tokens']

    choice = response.choices[0]
    message = choice.get("message")
    if message is not None:
        content = message.get("content")
        if content is not None:
              return {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "content": content
                }

    print("La respuesta no contiene la clave 'content'")
    return None


def build_message(visit):
    """
    This function takes a 'visit' parameter and generates a prompt message based on the given document.
    The prompt message includes instructions for identifying and enumerating test cases, their expected results,
    the component type they refer to, and the global functionality mentioned in the document.
    The function returns the extracted information in the form of a JSON string.

    Parameters:
    - visita: The document to be analyzed.

    Returns:
    - JSON string containing the extracted information.
    """

    prompt = f"""Te voy a dar un contexto y un documento, en base al documento: 
    - Identifica y enumeras TODOS los casos de prueba de testing de aplicaciones presentes (escritos) y su resultado esperado. Un ejemplo de caso de prueba es 'Validar que al darle clic en la opcion pasate un plan nos mande al la pantalla de "Pasate a un plan"' y su resultado esperado es 'Se debe mostrar la pantalla de "Pasate a un plan"'. Otro ejemplo de caso de prueba es 'Validar que al seleccionar el botón continuar permita avanzar a la pantalla de  check out' y su resultado esperado es 'Se debe mostrar la pantalla de check out'.
    - Identifica el tipo de componente de la aplicación al que hace referencia el caso de prueba (por ejemplo: 'botón continuar', 'pantalla', 'botón pasarme a un plan', 'Inicio de sesión', 'switch flujo de migraciones', 'parrillas', 'menú hamburguesa', 'campo RFC', 'Banner', 'Spinner', 'checkout', 'check box') y coloca este resultado en el campo 'componente'. 
    - Ten en cuenta que el componente tiene como máximo 5 palabras para ubicar la sección de la app, encambio el caso de prueba contiene una descripción más larga de la acción que hay que realizar.
    - Haz distinción de los casos que hablan del mantenedor y los que hablan de la app del usuario, coloca este resultado en el campo 'tipo'.
    - Además, debes identificar la funcionalidad global a la que hace referencia el texto completo, esta se encuentra generalmente al comienzo del documento. Por ejemplo: 'MANTENEDOR – SWITCH FLUJO DE MIGRACIONES-DESACTIVADO', 'MANTENEDOR – CONFIGURACIÓN DE PARRILLAS - SWITCH MOSTRAR EN EL APP – PLAN 2 – DESACTIVADO', 'MIGRACIONES – FILTRO / RANGO DE FECHAS' o descripciones similares. Este valor debes repetirlo para todos los casos de prueba que se encuentren en el documento y almacenarlo en el campo 'funcionalidad'. La funcionalidad ES IGUAL para todos los casos de prueba de un mismo documento, ignora la separación de mantenedor y app para el campo funcionalidad.
    - La salida debe ser SOLAMENTE un JSON con la informacion encontrada en el texto siguiendo la estructura: 
    {{ "1": {{
            "funcionalidad": extrae la funcionalidad y colocala aqui,
            "tipo": "mantenedor" o "aplicación",
            "componente": extrae el componente y colocalo aqui,
            "caso de prueba": extrae el caso de prueba y colocalo aqui,
            "resultado esperado": extrae el resultado esperado del caso de prueba y colocalo aqui,
            }},
        "2": {{
            "funcionalidad": extrae la funcionalidad y colocala aqui,
            "tipo": "mantenedor" o "aplicación",
            "componente": extrae el componente y colocalo aqui,
            "caso de prueba": extrae el caso de prueba y colocalo aqui,
            "resultado esperado": extrae el resultado esperado del caso de prueba y colocalo aqui,
             }},       
    }}
    - La salida debe ser un JSON que se pueda leer mediante json.loads() en python, incluye siempre los separadores correspondientes para que funcione la lectura. 
    Documento:{visit}"""

    message = [{"role": "user", "content": prompt}]
    
    return extraction(message)


def preprocess_docx(docx):
    """
    Preprocesses a docx file by splitting it into chunks based on the '#' character.
    
    Args:
        docx (str): The content of the docx file.
        
    Returns:
        tuple: A tuple containing the context (first chunk) and the documents (remaining chunks).
    """
    # Separa cada título del docx en un chunk:
    # Reemplazar '.\n-' por '#' para identificar cada título.
    # separar texto en chunks por caracter '#'. 
    docx_md = docx.replace('.\n-', '#')
    docx_md = docx_md.replace('\n–', '#')
    chunks_md = docx_md.split('#')
    context = chunks_md[0]
    documents = chunks_md[1:]
    return context, documents


def generate_response(documents):
    """
    Generates a response by processing a list of documents.

    Args:
        documentos (list): A list of documents to process.

    Returns:
        pandas.DataFrame: A DataFrame containing the generated results.
    """

    prompt_tokens = 0
    completion_tokens = 0

    # Inicializa un DataFrame vacío con las columnas deseadas.
    results = pd.DataFrame(columns=['funcionalidad','tipo','componente','caso de prueba', 'resultado esperado'])

    # Itera sobre cada chunk en 'documents'.
    for chunk in tqdm(documents):
        # Llama a 'build_message' con el chunk como argumento.
        result_dict = build_message(chunk)
        test_cases = result_dict['content']
        # Convierte la cadena de texto en un diccionario.
        #print(f"*****CASOs DE PRUEBA {chunk}******")
        #print(casos_de_prueba)
        test_cases_dict = json.loads(test_cases)
        prompt_tokens += result_dict['prompt_tokens']
        completion_tokens += result_dict['completion_tokens']
        # Convierte el diccionario anidado en un DataFrame y añádelo al DataFrame de resultados.
        for key, value in test_cases_dict.items():
            df = pd.DataFrame(value, index=[0])
            results = pd.concat([results, df], ignore_index=True)

    costo = f'El costo de la presente ejecución es u$s: {(prompt_tokens/1000)*0.003 + (completion_tokens/1000)*0.004}'
   
    # Ahora 'resultados' es un DataFrame que contiene todos los casos de prueba de todos los chunks.
    return results, costo


def generate_matrix(filename, mode='online'):
    if mode == 'local':
        from libs.docx_parser import getDoc
        docx = '../static/input/' + filename
        output_path = '../static/output/'
    else:
        from ai.libs.docx_parser import getDoc
        docx = 'static/input/' + filename
        output_path = 'static/output/'

    # Cargar el relative path del archivo que se quiere procesar.
    #path = dirname(dirname(dirname(abspath(__file__))))+'\\1.4 Datos\CP_Migraciones.docx'
  
    docx_file = getDoc(docx)
    
    context, document = preprocess_docx(docx_file)

    result, cost = generate_response(document)

    # Guarda los resultados en un archivo CSV que se lean ñ y tildes.
    #output_file = output_path  + 'resultados_generados.csv'
    #result.to_csv(output_file, index=False, encoding='utf-8-sig')

    # Guarda los resultados en un archivo Excel.
    output_file = output_path  + 'resultados_generados.xlsx'
    result.to_excel(output_file, sheet_name='Resultados', index=False)


    if mode == 'local':
        print(cost)
    else:
        msg = "Proceso terminado exitosamente (procesado: " + filename + ") puede consultar la matríz generada."
        cost = "El costo es de u$s 0.25." 
    
        return (cost, msg)


################# EL CODIGO DE ABAJO SE USA PARA CORRER LOCAL #####################
generate_matrix('cp_migraciones.docx', 'local')
