# File: search_snippets.py THIS FILE WORKS WITH LLAMAINDEX 6.0
import logging
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pinecone
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, StorageContext
from llama_index.vector_stores import PineconeVectorStore

# ================================================== prompts manager

from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

from llama_index.prompts.prompts import RefinePrompt, RefineTableContextPrompt

# Refine Prompt
CHAT_REFINE_PROMPT_TMPL_MSGS = [
    HumanMessagePromptTemplate.from_template("{query_str}"),
    AIMessagePromptTemplate.from_template("{existing_answer}"),
    HumanMessagePromptTemplate.from_template(
        "We have the opportunity to refine the above answer in the direction of putting attention to the non verbal and vitalistic part"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{context_msg}\n"
        "------------\n"
        "Given the new context, refine the original answer to better "
        "answer the question and put attention on the non verbal and the vitalistic part. "
        "If the context isn't useful, output the original answer again.",
    ),
]


CHAT_REFINE_PROMPT_LC = ChatPromptTemplate.from_messages(CHAT_REFINE_PROMPT_TMPL_MSGS)
CHAT_REFINE_PROMPT = RefinePrompt.from_langchain_prompt(CHAT_REFINE_PROMPT_LC)

# ======================================  end prompt manager


# Global Service context =========================================
from langchain.chat_models import ChatOpenAI
from llama_index import ServiceContext, LLMPredictor
from llama_index.callbacks import CallbackManager, LlamaDebugHandler

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

# Instantiate the model
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

# Instantiate the predictor with the model
llm_predictor = LLMPredictor(llm=llm)

# Instantiate a new ServiceContext with your preferred configuration
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size=512, callback_manager=callback_manager)

# Set the global service context
from llama_index import set_global_service_context
set_global_service_context(service_context)

# end global service context ======================================

# configure logging
logging.basicConfig(filename='search_snippets.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Set Pinecone API key and environment
pinecone_api_key = "93fa5151-3b05-430b-a7a1-78dcf32c7ed3"
pinecone_environment = "northamerica-northeast1-gcp"
pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

# Load Pinecone index
index_name = "project24"
pinecone_index = pinecone.Index(index_name)

# Create a Pinecone Vector Store
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# Load your documents
# Note: Replace './data' with the path to your data if necessary
documents = SimpleDirectoryReader('./data').load_data()

# Create a GPTVectorStoreIndex with the PineconeVectorStore
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = GPTVectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Define query engine with similarity_top_k
query_engine = index.as_query_engine(similarity_top_k=2)  # Modified

# set up Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# define search API endpoint
@app.route('/api/search', methods=['POST'])
# define search API endpoint
@app.route('/api/search', methods=['POST'])
def search_api():
    logging.info("function: search_api")
    logging.info(request.json)  # This will log the received data in your Flask server's console
    try:
        logging.info("Received search API request")
        query = request.json['query']
        logging.info(f'Querying index with "{query}"...')
        response = query_engine.query(query)  # receive a single response
        logging.info(f'Response is "{response}"...')
        snippet_text = str(response)  # convert response object to string
        logging.info(f'Snippet text is "{snippet_text}"')
        return jsonify({'query': query, 'response': snippet_text})  # return single response as a string
    except Exception as e:
        logging.error(f"Error searching for '{query}': {str(e)}")
        return jsonify({'error': str(e), 'error_message': f"Error searching for '{query}': {str(e)}"}), 500

# define search route
@app.route('/search', methods=['POST'])
def search_snippets():
    try:
        query = request.form['query']
        logging.info(f'Querying index with "{query}"...')
        response = query_engine.query(query)  # receive a single response
        logging.info(f'Answer is "{response}"...')
        return render_template('search_page.html', query=query, response=response)  # return single response
    except Exception as e:
        logging.error(f"Error searching for '{query}': {str(e)}")
        return render_template('search_page.html', query=query, snippets=[])

@app.route('/')
def search_page():
    return render_template('search_page.html')

@app.route('/simple_search')
def simple_search_page():
    return render_template('simple_search_page.html')

if __name__ == '__main__':
    logging.info('Starting server...')
    host = os.environ.get("FLASK_HOST", "127.0.0.1")

    # Set up console logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(message)s'))

    # Add the console handler to the root logger
    logging.getLogger().addHandler(console_handler)

    app.run(host=host, debug=True)
