#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import time

load_dotenv()

app = Flask(__name__)
CORS(app)

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))

target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
from constants import CHROMA_SETTINGS

embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = Chroma(persist_directory=persist_directory, 
            embedding_function=embeddings, 
            client_settings=CHROMA_SETTINGS)

retriever = db.as_retriever(search_kwargs={"k":target_source_chunks})

match model_type:
    case "LlamaCpp":
        llm = LlamaCpp(model_path=model_path, 
                       max_tokens=model_n_ctx, 
                       n_batch=model_n_batch, 
                       verbose=False)
    case "GPT4All":
        llm = GPT4All(model=model_path, 
                      max_tokens=model_n_ctx, 
                      backend='gptj', 
                      n_batch=model_n_batch, 
                      verbose=False)
    case _:
        raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

qa = RetrievalQA.from_chain_type(llm=llm, 
                                 chain_type="stuff", 
                                 retriever=retriever, 
                                 return_source_documents=True)

                                 
@app.route('/', methods=['POST'])
def process_query():
    IP = request.remote_addr
    query = request.json.get('query')
    start = time.time()
    res = qa(query)
    end = time.time()
    answer, docs = res['result'], res['source_documents']

    # Filter by Bitcoin keywords
    keywords = ["bitcoin", "btc", "satoshi", "blockchain",
                "hash", "sha-256", "proof of work", "digital signature",
                "bitcoin address", "block reward", "cryptocurrency",
                "private key", "public key", "wallet", "miner",
                "bitcoin transaction", "segwit", "lightning network",
                "coinbase transaction", "bitcoind", "utxo", "taproot",
                "bitcoin improvement proposal", "bip-", "byzantine"]

    if not any(keyword in answer.lower() for keyword in keywords):
        answer = "No bitcoin match found. Please consider uploading the relevant document to help train the model."
        source_docs = [{'source': "No sources found", 'content': ""}]
    else:
        source_docs = [{'source': doc.metadata["source"], 'content': doc.page_content} for doc in docs]

    result = {
        'query': query,
        'answer': answer,
        'time_taken': round(end - start, 2),
        'source_documents': source_docs
    }

    # Debug
    print(result)

    return jsonify(result)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
