import os
import PySimpleGUI as sg
from tqdm import tqdm
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
import tiktoken
from uuid import uuid4
from tqdm.auto import tqdm
import hashlib
import json

#Batch limit of upload size (can go upto 1000)
batch_limit = 100

#Helper function to calculae length of TOKENS not characters
def tiktoken_len(text):
    tiktoken.encoding_for_model('gpt-3.5-turbo')
    tokenizer = tiktoken.get_encoding('cl100k_base')

    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

#Splits each chunk of text by token length to help ChatGPT
def text_splitter():
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter

#Compute Hash
def compute_md5(text):
    m = hashlib.md5()
    m.update(text.encode('utf-8'))
    return m.hexdigest()

#Intializes the pinecone index and uploads the embeddings
##TODO: Split this function to improve readability
def initalize_embeddings(data, VERBOSE, progress_outer, window):
    texts = []
    metadatas = []
    model_name = 'text-embedding-ada-002'
    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=OPENAPI_KEY,
    )

    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT,
    )

    index = pinecone.GRPCIndex(PINECONE_INDEX_NAME)
    print(index.describe_index_stats()) if VERBOSE else None
    txt_splitter = text_splitter()
    texts = []
    metadatas = []
    status = "Injecting Metadata..."
    window['-OUTPUT-'].update(status)
    progress_outer.update_bar(0)
    for i, document in enumerate(tqdm(data)): #For each page in the document
        progress_outer.update_bar((i+ 1)/len(data)*100)
        metadata = {
            'source': document.metadata['source'],
            'page': document.metadata['page'] + 1,}
        record_texts = txt_splitter.split_text(document.page_content)
        record_metadatas = [{
            "chunk": j, "text": chunk, 'source': (document.metadata['source'].split('/')[-1] + ' Page: ' + str(document.metadata['page'])) 
        } for j, chunk in enumerate(record_texts)] #Each page will be associated with a metadata

        texts.extend(record_texts)
        metadatas.extend(record_metadatas)
    status = "Uploading to Pinecone..."
    window['-OUTPUT-'].update(status)
    for i in tqdm(range(0,len(texts),batch_limit)):
        progress_outer.update_bar((i+ 1)/len(texts)*100)
        text_tmp = texts[i:i+batch_limit]
        metadata_tmp = metadatas[i:i+batch_limit]
        ids = [compute_md5(text_tmp[i]) for i in range(len(text_tmp))]
        embeds = embed.embed_documents(text_tmp)
        index.upsert(vectors=zip(ids, embeds, metadata_tmp))
    progress_outer.update_bar(100)
# Function to perform some task
def perform_task(folder_path, openapi_key, pinecone_api_key, pinecone_environment, pinecone_index_name, progress_outer, window):
    # Simulated task with TQDM progress
    global OPENAPI_KEY
    OPENAPI_KEY = openapi_key
    global PINECONE_API_KEY 
    PINECONE_API_KEY = pinecone_api_key
    global PINECONE_ENVIRONMENT
    PINECONE_ENVIRONMENT = pinecone_environment
    global PINECONE_INDEX_NAME
    PINECONE_INDEX_NAME = pinecone_index_name
    status = "Reading files (this could take a while)..."
    window['-OUTPUT-'].update(status)
    progress_outer.update_bar(0)
    loader = DirectoryLoader(folder_path, glob="*.pdf", show_progress=True, loader_cls=PyPDFLoader)
    data = loader.load() #Data is an an array of Document objects with each object having a page_content and metadata
    #print(data)
    
    initalize_embeddings(data, True, progress_outer, window)

    # Close TQDM progress bars

# Define the layout of the GUI

#if default values exit, load them
try:
    with open("credentials.json", "r") as json_file:
        loaded_data = json.load(json_file)
except Exception as e:
    loaded_data = {
        "openapi_key": "",
        "pinecone_api_key": "",
        "pinecone_environment": "",
        "pinecone_index_name": "",
        "save_credentials": False
    }

layout = [
    [sg.Column([[sg.Image("img/logo.png")]], justification='center')],
    [sg.Text("Choose a folder:")],
    [sg.InputText(key="folder_path", size=(50,1), enable_events=True), sg.FolderBrowse()],
    [sg.Text("OPENAPI KEY:"), sg.InputText(key="openapi_key", size=(50,1), default_text=loaded_data["openapi_key"])],
    [sg.Text("PINECONE API KEY:", size=(None,None)), sg.InputText(key="pinecone_api_key", size=(46,1), default_text=loaded_data["pinecone_api_key"])],
    [sg.Text("PINECONE ENVIRONMENT:", size=(None,None)), sg.InputText(key="pinecone_environment", size=(40,1), default_text=loaded_data["pinecone_environment"])],
    [sg.Text("PINECONE INDEX NAME:", size=(None,None)), sg.InputText(key="pinecone_index_name", size=(42,1), default_text=loaded_data["pinecone_index_name"])],
    [sg.Checkbox("Save Credentials", key="save_credentials", default=loaded_data["save_credentials"])],
    [sg.ProgressBar(100, orientation="h", size=(50, 20), key="progress_outer")],
    [sg.Text("", size=(30, 1), key='-OUTPUT-')],
    [sg.Button("Upload Files"), sg.Button("Exit")],
    [sg.Column([[sg.Text("© 2023 Dev Masrani, Rohit Bathula")]], justification='right')]
]

# Create the window
window = sg.Window("StudyBuddy Ingestion Service", layout, resizable=True, finalize=True, size=(420, 580))
global status
status = ""
# Event loop
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED or event == "Exit":
        break
    elif event == "Upload Files":
        folder_path = values["folder_path"]
        openapi_key = values["openapi_key"]
        pinecone_api_key = values["pinecone_api_key"]
        pinecone_environment = values["pinecone_environment"]
        pinecone_index_name = values["pinecone_index_name"]
        save_credentials = values["save_credentials"]
        outval = {
            "openapi_key": values["openapi_key"],
            "pinecone_api_key": values["pinecone_api_key"],
            "pinecone_environment": values["pinecone_environment"],
            "pinecone_index_name": values["pinecone_index_name"],
            "save_credentials": values["save_credentials"]
        }
        if(save_credentials):
            with open('credentials.json', 'w') as outfile:
                json.dump(outval, outfile)
        else:
            os.remove("credentials.json")
        if folder_path:
            try:
                progress_outer = window["progress_outer"]
                perform_task(folder_path, openapi_key, pinecone_api_key, pinecone_environment, pinecone_index_name, progress_outer, window)
                sg.popup("Task completed successfully!", title="Success")
                break
            except Exception as e:
                sg.popup_error(f"An error occurred: {str(e)}", title="Error")
        else:
            sg.popup_error("Please choose a folder.", title="Error")
    window["folder_path"].update(values["folder_path"])  # Update the input field
    window.refresh()
# Close the window
window.close()
