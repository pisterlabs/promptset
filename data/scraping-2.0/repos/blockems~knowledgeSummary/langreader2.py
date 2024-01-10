import os
import json
from click import prompt

#Env Mgt
from dotenv import load_dotenv
from pypdf import PdfReader
#from torch import embedding


from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
data_conn = os.getenv('DATA_CONNECTION_STRING', './data/database.db')
data_processed= os.getenv('DATA_PROCESSED_DIR', './source/processed')
data_pre_processed= os.getenv('DATA_PRE_PROCESSED_DIR', './source/preprocessed')
data_source= os.getenv('DATA_SOURCE_DIR', './source')
data_directory= os.getenv('DATA_DIRECTORY', './data')
log_level = 'INFORMATION'

task = ["What are the key dates?",
    "What are the key deliverables?",
    "What are the key milestones?",
    "What are the key risks?",
    "What are the key assumptions?",
    "What are the key dependencies?",
    "What are the key issues?",
    "What are the key constraints?"]

def load_data():
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(embedding_function=embedding,persist_directory=data_directory)
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    docs = retriever.get_relevant_documents("What are the key dates?")

    print(docs)
    
def get_more():
    prompt = f'''ROLE:
    You are an experienced project manager

    CONTEXT: 
    You are reviewing a contract to extract important details.

    TASK:
    Review the content below and answer the following questions:
    {task[0]}

    CONTENT:
    {docs}'''

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {
                "role": "assistant",
                "content": prompt
            }
        ])
    print(response.choices[0].text)

if __name__ == '__main__':
    load_data()

