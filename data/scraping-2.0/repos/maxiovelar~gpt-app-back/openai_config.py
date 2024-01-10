import os
import time
import json
import requests
import nest_asyncio
from dotenv import load_dotenv

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import JSONLoader
from langchain.prompts import PromptTemplate

load_dotenv()

DATABASE_PATH = './db/'


################################################################################
# API Swell
################################################################################
url = os.getenv('SWELL_API_URL')

# Define your custom headers here
headers = {
    'Authorization': os.getenv('SWELL_AUTORIZATION_KEY'),  # If you have an API token or authentication
    'Content-Type': 'application/json',  # If needed for the request body
}

# Make the HTTP request with custom headers
response = requests.get(url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # Load the JSON data from the response
    json_data = response.json()

    # Initialize an empty list to store extracted data for each item
    extracted_data_list = []

    # Loop through each item in the array and extract the desired information
    for item in json_data["results"]:
        extracted_data = {
            "id": item.get("id", None),
            "name": item.get("name", None),
            "description": item.get("description", None),
            "price": item.get("price", None),
            "currency": item.get("currency", None),
            "stock": item.get("stock_level", None),
        }
        extracted_data_list.append(extracted_data)    
    
    open('data.json', 'w').write(json.dumps( extracted_data_list, indent=4))
    open('dataSwell.json', 'w').write(json.dumps( json_data, indent=4))
    # Process the JSON data as needed
    print(extracted_data_list)
else:
    print(f"Request failed with status code: {response.status_code}")

################################################################################
# Generate new Chroma instance
################################################################################
def get_chroma_instance():
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
    return  Chroma(embedding_function=embeddings, persist_directory=DATABASE_PATH)

################################################################################
# Read documents from JSON file and add them to Chroma instance
################################################################################
def revalidate():
    instance = get_chroma_instance()
    loader = JSONLoader(
        file_path='./data.json',
        jq_schema='.[]',
        text_content=False
    )

    if loader:
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators= ["\n\n", "\n", ".", ";", ",", " ", ""]) # se puede pasar regex a los separators
        texts = text_splitter.split_documents(documents)
        instance.add_documents(texts)

    instance.persist()

################################################################################
# Make OpenAI query
################################################################################
def query(query):
    instance = get_chroma_instance()
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Return an array with products.
    Don't return duplicated products.

    {context}

    Question: {question}
    Answer in JSON format:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain_type_kwargs = {"prompt": PROMPT}

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0,openai_api_key=os.getenv('OPENAI_API_KEY')),
        chain_type="stuff",
        retriever=instance.as_retriever(),
        chain_type_kwargs=chain_type_kwargs
    )

    res = qa.run(query)

    return json.loads(res)
