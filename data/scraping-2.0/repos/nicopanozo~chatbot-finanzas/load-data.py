# !pip install langchain
# !pip install python-dotenv
# !pip install openai
# !pip install pypdf
# pip install fnmatch

import os
import openai
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
import fnmatch
import csv

# Carga las variables de entorno
load_dotenv()

# Carga la API key de OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_data_pdf():

    # Obtiene la lista de archivos en el directorio
    pdf_files = os.listdir("./finances-data")

    # Filtra los archivos PDF
    pdf_files = fnmatch.filter(pdf_files, "*.pdf")

    # Lista para almacenar el contenido de los PDF
    all_pdf_content = []

    # Itera sobre los archivos PDF
    for pdf_file in pdf_files:
        # Carga el archivo PDF
        loader = PyPDFLoader("./finances-data/" + pdf_file)
        pages = loader.load()

        # Agrega el contenido del archivo PDF a la lista
        all_pdf_content.append(pages)

    return all_pdf_content


# Llama a la función
# all_pdf_content = load_data_pdf()
# print(all_pdf_content)

def load_csv_data():
    """
    Carga el contenido de todos los archivos CSV en una sola variable.

    Devuelve:
        Una lista de listas, donde cada lista contiene el contenido de una fila de todos los archivos CSV.
    """

    # Obtiene la lista de archivos en el directorio
    files = os.listdir("./finances-data")

    # Filtra los archivos CSV
    csv_files = fnmatch.filter(files, "*.csv")

    # Lista para almacenar el contenido de todos los CSV
    all_data = []

    # Itera sobre los archivos CSV
    for csv_file in csv_files:
        # Carga el archivo CSV
        data = csv.reader(open(f"./finances-data/{csv_file}", "r"), delimiter=",")

        # Agrega el contenido del archivo CSV a la lista
        all_data.extend(data)

    return all_data


# Llama a la función
all_data = load_csv_data()

# print(all_data)
# Llama a las funciones
pdf_data = load_data_pdf()
csv_data = load_csv_data()

# Guarda los resultados en los archivos de texto
def save_pdf_data():

    # Abre el archivo de texto
    with open("after-load/pdf.txt", "w") as f:
        # Convierte la lista a una cadena
        data_string = str(pdf_data)

        # Escribe la cadena en el archivo
        f.write(data_string)
        f.write("\n")


def save_csv_data():

    # Abre el archivo de texto
    with open("after-load/csv.txt", "w") as f:
        # Convierte la lista a una cadena
        data_string = str(csv_data)

        # Escribe la cadena en el archivo
        f.write(data_string)
        f.write("\n")


# Llama a las funciones
save_pdf_data()
save_csv_data()