from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper, ServiceContext
from llama_index import load_index_from_storage, StorageContext
from llama_index.storage.storage_context import StorageContext

from langchain import OpenAI
import os

from flask import Flask, request, jsonify
from flask_cors import CORS
import openai


app = Flask(__name__)
CORS(app) 
#asadsd

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
openai.api_key = os.getenv('OPENAI_API_KEY')


def build_storage(data_dir):
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo"))

    max_input_size = 8000

    num_output = 2000

    max_chunk_overlap = 0

    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    documents = SimpleDirectoryReader('data').load_data()

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    # service_context==service_context
    index.storage_context.persist()

    return index

def read_from_storage(persist_dir):
    print("here12")
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    return load_index_from_storage(storage_context)


@app.route('/query', methods=['POST'])
def query():
    persist_dir = "./storage"
    data_dir = "./data"
    index = None
    print('here2')
    if os.path.exists(persist_dir):
        index = read_from_storage(persist_dir)
    else:
        index = build_storage(data_dir)
        

    data = request.get_json()
    question = data.get('question')
    query_engine = index.as_query_engine()
    response  = query_engine.query(question)
    response = str(response)

    return jsonify({'response': response})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
