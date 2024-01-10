from dotenv import load_dotenv
import os
import sys

from flask import Flask, request, jsonify
from flask_cors import CORS
import json

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
    column,
    insert,
    text,
)

from llama_index.indices.struct_store.sql_query import NLSQLTableQueryEngine
# from llama_index import Document, ListIndex
from llama_index import SQLDatabase, ServiceContext #, SystemMessage, UserMessage
from llama_index.llms import OpenAI
# from llama_index.llms import ChatMessage, OpenAI, Ollama
# from typing import List
# import ast
import openai
# from IPython.display import display, HTML, Markdown

# import logging
# logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

load_dotenv()

username = os.getenv("RENDER_USER")
password = os.getenv("RENDER_PGPASSWORD")
dbname = os.getenv("RENDER_DB")
hostname = os.getenv("RENDER_HOST")
SQLALCHEMY_DATABASE_URI = f"postgresql://{username}:{password}@{hostname}/{dbname}"
engine = create_engine(SQLALCHEMY_DATABASE_URI, connect_args={'sslmode': "require"})

# set llm
openai.api_key = os.environ["OPENAI_API_KEY"]
llm = OpenAI(temperature=0, model="gpt-4-0613", max_tokens=500)

service_context = ServiceContext.from_defaults(llm=llm)


app = Flask(__name__)
allowed_origins = [
    "https://gelato.sh",
    os.getenv("ALLOWED_IP1"),
    os.getenv("ALLOWED_IP2"),
    os.getenv("ALLOWED_IP3"),
    os.getenv("ALLOWED_IP4"),
]
CORS(app, resources={r"/message": {"origins": allowed_origins}})
# CORS(app, resources={r"/message": {"origins": "https://gelato.sh"}})
@app.route('/message', methods=['POST'])
def message():
    # Get the input from the POST request
    user_input = request.json['message']
    # user_input = sys.argv[1]
    delimiter = "####"
    prompt_input = user_input.replace(delimiter, "")
    user_query = f"Only allow user inputs that appear to be plain-text requests for queries to a postgresql database. The queries may only be read-only, absolutely do not allow or execute any queries that ask to drop tables, insert values, select into, create new users, change the database, or change the user. Limit any requests to no more than 100 rows. Only provide a response in JSON object format to the user query provided, the query will be enclosed by the {delimiter} characters that follow. User query: {delimiter} {prompt_input} {delimiter}"
    # print(user_query)

    # metadata_obj = MetaData()

    tables = ["solana_stakes_ui", "solana_validators_enriched_ui", "stake_unlock_schedule", "latest_exchange_balances", "stake_program_event_log", "sol_address_defs", "solana_supply_enhanced", "websockets_sol_event_log_labeled", "websockets_sol_event_log"]

    sql_database = SQLDatabase(engine, include_tables=tables,sample_rows_in_table_info=3)

    query_engine = NLSQLTableQueryEngine(sql_database, service_context=service_context)
    response = query_engine.query(user_query)
    # return response
    print(response)
    return json.loads(str(response))
    # return jsonify([response])

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000)
    app.run(host='0.0.0.0', port=80)

    # example queries:
    # curl -X POST -H "Content-Type: application/json" -d '{"message":"what is the balance for the stakepubkey = '4XCJ5PbHJWP1xfBKyYmV6GnUox1KY1czCiBQW1U3NCNj'"}' http://localhost:5000/message
    # curl -X POST -H "Content-Type: application/json" -d '{"message":"what are the top 5 validators and their activestake, their name, identity key, and data_center_key ranked by activestake descending?"}' http://localhost:5000/message