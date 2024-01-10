#!/usr/bin/env python3
from constants import CHROMA_SETTINGS
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import argparse

app = Flask(__name__)
CORS(app)


@app.route('/api/start', methods=['POST', 'GET', 'PUT'])
def start_endpoint():
    # # Remaining code for question-answering goes here
    # # Parse the command line arguments
    response = {'data': "Enter a query"}
    return jsonify(response)


load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))
# Added a paramater for GPU layer numbers
n_gpu_layers = os.environ.get('N_GPU_LAYERS') 
# #Path to Cuda installation
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin")
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/extras/CUPTI/lib64")
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/include")
os.add_dll_directory("C:/Tools/cuda/bin")
args = ("")
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = Chroma(persist_directory=persist_directory,
            embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
print("Loading the retriever...")
retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
# activate/deactivate the streaming StdOut callback for LLMs
callbacks = [StreamingStdOutCallbackHandler()]
print("Loading the LLM...")
# Prepare the LLM
match model_type:
    case "LlamaCpp":
        llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False, n_gpu_layers=n_gpu_layers)
    case "GPT4All":
        llm = GPT4All(model=model_path, n_ctx=model_n_ctx,
                      backend='gptj', callbacks=callbacks, verbose=False)
    case _default:
        print(f"Model {model_type} not supported!")
        exit
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

# Interactive questions and answers


@app.route('/api/run', methods=['POST', 'GET', 'PUT'])
def run_endpoint():
    try:
        response = {}
        source = str("")
        input = request.get_json('query')
        query = input['query']
        print(query)
        if query == "exit":
            return jsonify({'data': "Bye!"})

        # Get the answer from the chain
        res = qa(query)
        answer, docs = res['result'], [
        ] if 1 == 0 else res['source_documents']

        # Print the result
        print("\n\n> Question:")  # Remove this one API is implemented
        print(query)  # Change this to send query over api
        print("\n> Answer:")  # Remove this one API is implemented
        print(answer)  # Change this to send answer over api

        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)
            source += document.page_content
        # Add the page content to the response dictionary
        response.update({'docs': source})
        # Add another key-value pair to the response dictionary
        response.update({'data': answer})
        return jsonify(response)
    except Exception as e:
        print(e)
        return jsonify({'data': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
