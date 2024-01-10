#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse
import time

# for API
import requests as requests
from flask import Flask, request
import json

load_dotenv('../.env')

api_host=os.environ.get('API_HOST')
api_port=os.environ.get('API_PORT')
client_host=os.environ.get('CLIENT_HOST')
client_port=os.environ.get('CLIENT_PORT')

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX') # this is the number of tokens to feed to the model
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8)) # this is the number of chunks to split the input sequence into
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4)) # this is the number of source chunks to retrieve for each query

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'


from constants import CHROMA_SETTINGS

qa, args = None, None
queue = []


def main():
    global qa, args
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case _default:
            # raise exception if model_type is not supported
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")
        
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)

    # Start the API
    app.run(api_host,api_port)


@app.route('/query', methods=[ 'POST'])
def ask():
    query = request.json
    queue.append(query)

    if len(queue) == 1:
        process_query(queue[0])

    return 'success'



def process_query(query):
    print("\n\n> New query received: " + query['question'])

    query['answer'] = 'started processing'
    pushQuery(query)

    # Get the answer from the chain
    question = query['question']
    start = time.time()
    res = qa(question)
    answer, docs = res['result'], [] if args.hide_source else res['source_documents']
    end = time.time()

    # Print the result
    print("\n\n> Question:")
    print(question)
    print(f"\n> Answer (took {round(end - start, 2)} s.):")
    print(answer)

    # Print the relevant sources used for the answer
    for document in docs:
        print("\n> " + document.metadata["source"] + ":")
        print(document.page_content)

    # Push the answer to the client
    query['answer'] = answer
    # query['sources'] = docs
    sources = []
    for document in docs:
        sources.append({
            'title': document.metadata["source"],
            'content': document.page_content
        })
    query['sources'] = sources
    pushQuery(query)

    # Remove the query from the queue and process the next one
    queue.pop(0)
    if len(queue) > 0:
        process_query(queue[0])


def pushQuery(query):
    query['timeline'].append(time.time()*1000)
    headers = {
        'Content-Type': 'application/json'
    }
    url = "http://"+client_host+":"+client_port+"/pushAnswer"
    payload = json.dumps(query)
    requests.request("POST", url, headers=headers, data=payload)
    return



def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
