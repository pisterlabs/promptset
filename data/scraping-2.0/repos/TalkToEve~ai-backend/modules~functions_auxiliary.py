import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
# Obtain the parent directory
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)

from langchain.text_splitter import CharacterTextSplitter

from configurations.config import CHUNK_OVERLAP, CHUNK_SIZE

def obtain_number_session(file_name):
    parts = file_name.split("_")
    return int(parts[2])

def read_txt_files(path):
    file_path  = path 
    with open(file_path, 'r') as output_file:
        content = output_file.read()
    return content


def split_txt_file(raw_text, separator = "\n#"):
    text_splitter = CharacterTextSplitter(
            separator=separator,
            chunk_size= CHUNK_SIZE,
            chunk_overlap= CHUNK_OVERLAP,
            length_function=len,
            
            
        )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def convert_messages_in_txt(conversation_messages):
    conversation_txt = ""
    for message in conversation_messages:
        if message.__class__.__name__ == "AIMessage":
            conversation_txt += 'Eve: ' + message.content + '\n'
        elif message.__class__.__name__ == "HumanMessage":
            conversation_txt += 'Patient: ' + message.content + '\n'
    return conversation_txt

def save_as_txt(path, content):
    file_path  = path 
    if not os.path.exists(os.path.dirname(file_path)):
        #Create the directory
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, 'w') as output_file:
        output_file.write(content)
        
def read_txt_file(path):
    file_path  = path 
    with open(file_path, 'r') as output_file:
        content = output_file.read()
    return content
        
def convert_state_str_to_variables(state_str):
    lineas = state_str.split('\n')

    opciones = {}  # Crear un diccionario para almacenar las opciones

    for linea in lineas:
        if ":" in linea:
            clave, valor = linea.split(':')
            opciones[clave.strip()] = valor.strip() == "True"

    # Encontrar las claves con valores "True"
    claves_true = [clave for clave, valor in opciones.items() if valor]

    return claves_true[0] 