import streamlit as st
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import pathlib
import PyPDF2
from pdf2image import convert_from_path
from PIL import Image


# Configuración de Azure Blob Storage


# Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://arkano-openai-dev.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = os.getenv("OPENAI_API_KEY")

azure_connection_string = os.getenv("AZURE_ENDPOINT")
container_name = "raw"

def get_blob_url(blob_name):
    blob_service_client = BlobServiceClient.from_connection_string(azure_connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    return blob_client.url



def upload_to_azure_blob(file_path, blob_name):
    blob_service_client = BlobServiceClient.from_connection_string(azure_connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    
    with open(file_path, "rb") as data:
        container_client.upload_blob(blob_name, data, overwrite=True)


def consulta_AI(texto):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt='Leer este contrato en linea: '+ str(texto)+' y sugerir no mas de 4 clausulas adicionales que no estén cubiertas? tu respuesta en formato de lista \n' ,
        temperature=0.60,
        max_tokens=1704,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        best_of=1)
    return response


def proceso():
    st.title("OPENAI Demo - CAMPOSOL - Contratos")
    
    uploaded_file = st.file_uploader("Selecciona un archivo PDF", type=["PDF","DOCX"])
    if uploaded_file is not None:
        st.write("Archivo cargado:", uploaded_file.name)

    if btn_0:
        bytes_data = uploaded_file.getvalue()
        data = uploaded_file.getvalue()

        parent_path = pathlib.Path(__file__).parent.parent.resolve()
        save_path = os.path.join(parent_path, "data")
        complete_name = os.path.join(save_path, uploaded_file.name)
        with open(complete_name, "wb") as f:
            f.write(bytes_data)
        st.session_state["upload_state"] = complete_name

        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
            
        blob_name = uploaded_file.name
        upload_to_azure_blob(file_path, blob_name)
        blob_url = get_blob_url(blob_name)
        st.session_state["blob_url"] = blob_url
        st.success("Archivo cargado exitosamente")

    if btn_1:
        # if st.session_state["texto"] != '' and st.session_state["texto"] is not None:
        if st.session_state["blob_url"] != '' and st.session_state["blob_url"] is not None:
            # texto = str(st.session_state["texto"])
            respuesta = consulta_AI(st.session_state["blob_url"])
            st.info("Las clausulas sugeridas al contrato son:")
            st.success(respuesta.choices[0].text)
            st.success("Archivo analizado exitosamente")
            st.write(respuesta)
            st.write("Nombre del archivo:", uploaded_file.name)

def main():
    
    proceso()
col1,col2,col3 = st.columns(3)
container = st.container()
image = Image.open('logo.png')
with container:
    col2.image(image, use_column_width=True, width=200)

btn_0 = st.sidebar.button("Guardar")
btn_1 = st.sidebar.button("Analizar")

if __name__ == "__main__":
    main()

