from IPython.display import Markdown, display
from vertexai.preview.language_models import (ChatModel, InputOutputTextPair,
                                              TextEmbeddingModel,
                                              TextGenerationModel)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
import os
import time


# function to load docs from path
def load_docs_from_path(repo_path):
    docs = []
    for dirpath, dirnames, filenames in os.walk(repo_path):
        for file in filenames:
            try: 
                loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                docs.extend(loader.load_and_split())
            except Exception as e: 
                print(e)
                pass
    return docs

# function to generate splitted text 

def split_text(docs, chunk_size=1000, chunk_overlap=0):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(docs)
    return texts

# path to the dataset
dataset_path = 'local_vector_db/langchain'

# google embedding
from langchain.embeddings import VertexAIEmbeddings
import json
from google.oauth2 import service_account
import google.cloud.aiplatform as aiplatform
import vertexai


def init_platform():
    with open("app/service_account.json") as f:  # replace 'serviceAccount.json' with the path to your file if necessary
        service_account_info = json.load(f)

    my_credentials = service_account.Credentials.from_service_account_info(
        service_account_info
    )

    # Initialize Google AI Platform with project details and credentials
    aiplatform.init(
        credentials=my_credentials,
    )

    with open("app/service_account.json", encoding="utf-8") as f:
        project_json = json.load(f)
        project_id = project_json["project_id"]

    # Initialize Vertex AI with project and location
    vertexai.init(project=project_id, location="us-central1")


if __name__ == "__main__":
    try:
        init_platform()
    except Exception as e:
        print(e)
    # read langchain into vector
    docs = load_docs_from_path("langchain")
    texts = split_text(docs)

    print("Number of texts: ", len(texts))
    embeddings = VertexAIEmbeddings()

    db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

    group_size = 1000
    group_num = len(texts) // group_size
    try:
        db.from_documents(documents=texts, embedding=embeddings, overwrite=True)
    except Exception as e:
        print(e)
    # for i in range(group_num):
    #     db.add_texts(texts[i * group_size: (i + 1) * group_size])
    #     time.sleep(60)