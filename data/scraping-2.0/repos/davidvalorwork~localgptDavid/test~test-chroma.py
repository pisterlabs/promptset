import sys
sys.path.append('../')

from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
import torch
import json
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDINGS = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})

DB = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=EMBEDDINGS,
    client_settings=CHROMA_SETTINGS,
)

query = "Que nombremientos tiene Jose Sosa Briceño" # Bien
query = "Jose Sosa Briceño" # Bien 
query = "Dame nombramientos para Jose Briceño" # Mal
docs = DB.similarity_search(query)

print("Number of docs found: ", len(docs))

json_1_doc = json.loads(docs[0].json())

print(json_1_doc['page_content'])
