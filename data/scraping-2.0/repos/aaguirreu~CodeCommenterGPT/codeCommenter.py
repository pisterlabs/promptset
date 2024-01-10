import sys
import openai
import os
import json
import re
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configurar la API de OpenAI con tu clave de API
openai.api_key = os.getenv('OPENAI_API_KEY')

def leer_archivo_json(nombre_archivo):
    # Lee el archivo JSON
    with open(nombre_archivo, 'r') as archivo:
        contenido = json.load(archivo)
    
    # Retorna el contenido del archivo JSON
    return contenido

def gpt_request(sql_code):
    # Comentar el código SQL explicando lo que hace
    programming_language = 'SQL'
    language = 'Spanish'
    messages = leer_archivo_json('context.json')
    messages.append({
            "role": "user",
            "content": f"Correct. Now, do the same with the next {programming_language} code. Write all comments in {language} language:\n{sql_code}"
            })

    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
        )
    
    return chat_completion.choices[0].message.content

def obtener_numero(cadena):
    if "[" in cadena and "-" in cadena and "]" in cadena:
        inicio = cadena.index("[") + 1
        fin = cadena.index("-")
        numero = cadena[inicio:fin]
    elif "[" in cadena and "]" in cadena:
        inicio = cadena.index("[") + 1
        fin = cadena.index("]")
        numero = cadena[inicio:fin]
    else:
        return None
    if numero.isdigit():
        return int(numero)
    
    return None

def agregar_comentarios(fragmento_codigo, comentarios):
    lineas_codigo = fragmento_codigo.split('\n')
    if type(lineas_codigo) is not list: return
    comentarios = comentarios.split('\n')
    # Recorrer los comentarios en orden descendente
    for comentario in reversed(comentarios):
        if not comentario.startswith('['): continue
        print(f"*\n{comentario}\n")
        comentario_strip = comentario.strip("] ")

        # Si el comentario no tiene un número de línea, continuar
        num_linea = ""
        if "]" in comentario_strip:
            # separar el número de línea del comentario
            num_linea, comentario = comentario.split("]", 1)
        else: continue

        # Obtener el número de línea del comentario
        num_linea = num_linea+']'
        num_linea = obtener_numero(num_linea)

        # Verificar si el comentario tiene un número de línea válido
        if num_linea is None: continue

        # Agregar el comentario en la línea correspondiente
        comentario = f'--{comentario}'
        lineas_codigo.insert(num_linea-1, comentario)
    # Unir las líneas de código nuevamente
    codigo_actualizado = '\n'.join(lineas_codigo)
    return codigo_actualizado

def recorrer_archivos(file_path, sql_code):
    # Fragmentar el código SQL en fragmentos de tamaño fijo
    fragment_size = 2000  # Tamaño máximo de fragmento en tokens
    fragments = [sql_code[i:i+fragment_size] for i in range(0, len(sql_code), fragment_size)]

    # Comentar cada fragmento del código SQL y guardar las respuestas en el archivo
    output_file_path = file_path
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        remaining_line = ''
        for i, fragment in enumerate(fragments):
            # Combinar la línea restante del fragmento anterior con el fragmento actual
            fragment = remaining_line + fragment
            remaining_line = ''

            # Verificar si la última línea del fragmento actual queda cortada
            lines = fragment.split('\n')
            if len(lines) > 1 and not lines[-1].endswith('--'):
                # La última línea queda cortada, guardarla para el siguiente fragmento
                remaining_line = lines[-1]
                fragment = '\n'.join(lines[:-1])

            fragment = fragment.split('\n')
            fragment_with_indexes = []

            # Agregar el número de línea a cada línea del fragmento
            for j, line in enumerate(fragment, start=1):
                fragment_with_indexes.append(f"{j} {line}")
            
            fragment = '\n'.join(fragment)
            fragment_with_indexes = '\n'.join(fragment_with_indexes)
            #print(fragment_with_indexes)
            comments = gpt_request(fragment_with_indexes)
            commented_code = agregar_comentarios(fragment, comments)
            print(f'-- Respuesta {i+1}:\n{commented_code}')
            output_file.write(f'\n{commented_code}')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Debe proporcionar la dirección de la caperta con archivos .sql como argumento.')
    else:
        folder_path = sys.argv[1]
        output_folder_path = os.path.join(os.path.dirname(folder_path), "pgSQL_commented")
        # Verificar si la carpeta de destino existe, si no, crearla
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        
        # Obtener la lista de archivos .sql en la carpeta de origen
        archivos_sql = [archivo for archivo in os.listdir(folder_path) if archivo.endswith(".sql")]

        for archivo in archivos_sql:
            file_path = os.path.join(folder_path, archivo)
            output_file_path = os.path.join(output_folder_path, archivo)
            # Verificar si el archivo ya existe en la carpeta de destino
            if not os.path.exists(output_file_path):
                print(file_path)
                with open(file_path, 'r', encoding='utf-8') as sql_file:
                    sql_code = sql_file.read()

                print(f'Comentando el archivo {archivo}...')
                recorrer_archivos(output_file_path, sql_code)
