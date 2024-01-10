import os
os.environ["FLASK_ENV"] = "development"

dependencies = [
   'pip install -q torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118',
   'pip install -q torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118',
   'pip install -q torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118',
   'pip install -q transformers==4.34.0',
   'pip install -q langchain==0.0.326',
   'pip install -q flask==3.0.0',
   'pip install -q pypdf==3.17.0',
   'pip install -q cython==3.0.5',
   'pip install -q sentence_transformers==2.2.2',
   'pip install -q chromadb==0.4.15',
   'pip install -q accelerate==0.23.0',
   'pip install -q sentencepiece==0.1.99',
   'pip install -q pyngrok==7.0.0',
   'pip install -q gdown==4.7.1'
   ]

for command in dependencies:
    os.system(command)

# ---------------------------------------------------
# creating the configuration script
# ---------------------------------------------------
config_str = '''
{
   "device_map": {
    "cuda:0": "20GiB",
    "cuda:1": "20GiB",
    "cpu": "30GiB"
    },
    "required_python_version": "cp311",
    "models": [
        {
            "key": "information_source.zip",
            "name": "information_source",
            "access_token": "https://drive.google.com/uc?id=1O5gQKwcYA_7JzQr8JmhmonVwTov_cWlL"
        },
        {
            "key": "BAAI/bge-small-en",
            "name": "embeddings_model",
            "access_token": "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"
        },
        {
            "key": "meta-llama/Llama-2-7b-chat-hf",
            "name": "llama_model",
            "access_token": "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"
        }

    ],
    "functions": [
        {
            "name": "QueryContent",
            "description": "When a user queries something, the AI first search into the chroma database to search for answers",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Answer to the query to search from the database"
                    }
                },
                "required": ["prompt"]
            },
            "input_type": "json",
            "return_type": "application/json"
        }
    ]
}
'''

# ---------------------------------------------------
# importing the reqiored libraries
# ---------------------------------------------------
import json
import time
import gdown
import torch
import zipfile
import textwrap
import requests
import threading

from pyngrok import ngrok
from flask import Flask, request, jsonify

from transformers import pipeline

from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PROMPT_CONFIG:
    def __init__(self):
        # ---------------------------------------------------
        # forming the LLaMA-2 prompt style
        # ---------------------------------------------------
        self.B_INST, self.E_INST = "[INST]", "[/INST]"
        self.B_SYS, self.E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        self.DEFAULT_SYSTEM_PROMPT = """\
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

        self.SYS_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided. Your answers should only answer the question once and not have any text after the answer is done.
        ​
        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. """

        self.INSTRUCTION = """CONTEXT:/n/n {context}/n
        ​
        Question: {question}"""

        SYSTEM_PROMPT = self.B_SYS + self.DEFAULT_SYSTEM_PROMPT + self.E_SYS
        self.prompt_template =  self.B_INST + SYSTEM_PROMPT + self.INSTRUCTION + self.E_INST

        llama_prompt = PromptTemplate(template=self.prompt_template, input_variables=["context", "question"])
        self.chain_type_kwargs = {"prompt": llama_prompt}

constants = PROMPT_CONFIG()

class DataManager:
    def __init__(self, key, name, token):
        extension = key.split(".")[-1]
        self.file = f"{name}.{extension}"
        gdown.download(token, output=self.file, quiet=False, fuzzy=True)

        #extract file
        with zipfile.ZipFile(self.file, 'r') as zip_ref:
            zip_ref.extractall(name)

# ---------------------------------------------------
# chat completion functions
# ---------------------------------------------------
def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    result = wrap_text_preserve_newlines(llm_response['result'])
    query = llm_response['query']
    unique_sources = set()

    for source in llm_response["source_documents"]:
        unique_sources.add(source.metadata['source'])

    sources = '\n\nSources: '
    for source in unique_sources:
        
        sources = sources + source
    
    return {"query": query, "result": result, "sources": sources}
        

# ---------------------------------------------------
# create the model manager class
# ---------------------------------------------------
class ModelManager:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.device = self.select_device()

    def select_device(self):
        if not torch.cuda.is_available():
            return "cpu"

        device_map = self.config.get('device_map', {})
        available_devices = list(device_map.keys())
        return available_devices[0] if available_devices else "cpu"

    def setup(self):
        self.models.clear()

        # form the model in sync for the query retrieval
        for model_info in self.config["models"]:

            if model_info["name"] == 'information_source':
                DataManager(model_info["key"], model_info["name"], model_info["access_token"])
            
            # get the embeddings model to embed the pdfs in the folder
            if model_info["name"] == 'embeddings_model':
                loader      = DirectoryLoader(f"{self.config['models'][0]['name']}", glob="./*.pdf", recursive=True, loader_cls=PyPDFLoader)
                documents   = loader.load()


                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = text_splitter.split_documents(documents)
                
                embed_model     = HuggingFaceBgeEmbeddings(model_name=model_info["key"], 
                                                           model_kwargs={'device': self.device}, 
                                                           encode_kwargs={'normalize_embeddings': True},
                                                           query_instruction="Generate a representation for this sentence for retrieving related articles: ")

                vectordb        = Chroma.from_documents(documents=texts, embedding=embed_model, persist_directory='db')
                retriever       = vectordb.as_retriever(search_kwargs={"k": 5})

                self.models[model_info["name"]] = retriever

            elif model_info['name'] == 'llama_model':
                torch.cuda.empty_cache()
  
                #int(self.select_device()[-1])
                pipe = pipeline("text-generation", model=model_info["key"], max_length=2048, temperature=0.75, top_p=0.95, repetition_penalty=1.2, token=model_info["access_token"])
               
                llm  = HuggingFacePipeline(pipeline=pipe)
               
                qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=self.models['embeddings_model'], chain_type_kwargs=constants.chain_type_kwargs, return_source_documents=True)
    
                self.models[model_info["name"]] = qa_chain
              

                    

        return True


    def infer(self, parameters):
        try:
            ### BEGIN USER EDITABLE SECTION ###
            query_model = self.models["llama_model"]
            
            llm_response = query_model(parameters['prompt'])
            torch.cuda.empty_cache() if self.device != "cpu" else None
            #llm_response = process_llm_response(llm_response)
            return llm_response
            ### END USER EDITABLE SECTION ###
        except Exception as e:
            print(f"Error during inference: {e}")
            return None

# ---------------------------------------------------
# load configurations for the program block
# ---------------------------------------------------
config          = json.loads(config_str)
model_manager   = ModelManager(config)

# ---------------------------------------------------
# make the ap and the corresponding function
# ---------------------------------------------------
app     = Flask(__name__)


# startup the application
# ---------------------------------------------------
@app.route('/setup', methods=['GET'])
def setup():
    model_manager.setup()
    return jsonify({"status": "models loaded successfully"})

@app.route('/<function_name>', methods=['POST'])
def generic_route(function_name):
    function_config = next((f for f in config["functions"] if f["name"] == function_name), None)

    if not function_config:
        return jsonify({"error": "Invalid endpoint"}), 404

    if function_config["input_type"] != "json":
        return jsonify({"error": f"Unsupported input type {function_config['input_type']}"}), 400

    data = request.json
    parameters = {k: data[k] for k in function_config["parameters"]["properties"].keys() if k in data}

    result = model_manager.infer(parameters)

    result = process_llm_response(result)
    if result:
        return jsonify(result), 200
        #return app.response_class(result, content_type=function_config["return_type"])
    else:
        return jsonify({"error": "Error during inference"}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    # Generic exception handler
    return jsonify(error=str(e)), 500


# Start the Flask server in a new thread

threading.Thread(target=app.run, kwargs={"use_reloader": False}).start()

# Set up Ngrok to create a tunnel to the Flask server

public_url = ngrok.connect(5000).public_url
function_names = [func['name'] for func in config["functions"]]
print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{5000}/\"")

# Loop over function_names and print them

for function_name in function_names:

    time.sleep(5)

    print(f'Endpoint here: {public_url}/{function_name}')

BASE_URL = f"{public_url}"

### BEGIN USER EDITABLE SECTION ###

def setup_test():

    response = requests.get(f"{BASE_URL}/setup")
    # Check if the request was successful
    if response.status_code == 200:

        return (True, response.json())  # True indicates success

    else:

        return (False, response.json())  # False indicates an error

def infer_test(prompt="Tell me about music"):

    # create prompt
    prompt = "Question: " + prompt + " Answer:"

    headers = {

        "Content-Type": "application/json"

    }

    data = {
        "prompt": prompt
    }

    response = requests.post(f"{BASE_URL}/QueryContent", headers=headers, json=data)

    if response.status_code == 200:
        # Save the image to a file
        
        dict_response = response.json()
        with open("output_text.txt", "w") as file:
            
            file.write(str(dict_response['result']))
            file.write(str(dict_response['sources']))

        print("Answer saved as output_text.txt!")

        return (True, response.json())  # True indicates success

    else:

        return (False, response.json())  # False indicates an error


def infer_test_url(prompt="which city is this?"):

    # create promt

    prompt = "Question: " + prompt + " Answer:"

    headers = {

        "Content-Type": "application/json"

    }

    data = {

        "prompt": prompt

    }

    response = requests.post(f"{BASE_URL}/QueryContent", headers=headers, json=data)
    if response.status_code == 200:
        dict_response = response.json()
        with open("output_text.txt", "w") as file:

            file.write(str(dict_response['result']))
            file.write(str(dict_response['sources']))


        print("Answer saved as output_text.txt!")

        return (True, response.json())  # True indicates success

    else:

        return (False, response.json())  # False indicates an error

### END USER EDITABLE SECTION ###

# Testing

result_setup = setup_test()
result_infer = infer_test()

print(result_infer)

result_infer_url = infer_test_url("Tell me about books")

print(result_infer_url)
