from flask import Flask, jsonify, request
from llama_index import (
    Document,
    GPTVectorStoreIndex,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
)
from llama_index.node_parser import SimpleNodeParser
from langchain import OpenAI
import os
from typing import List
from llama_index.data_structs.node import Node
import pyodbc
from sqlalchemy import create_engine, MetaData, Table, select
import json


# initialize our Flask application
app = Flask(__name__)


@app.route("/chat_bot_response", methods=["GET"])
def chat_bot_response():
    data = request.get_json()
    input = data["input"]
    print(f"chat bot input:\n {input}")

    # get keys
    open_ai_dict = get_value_from_json("secrets.json", "open_ai")
    db_dict = get_value_from_json("secrets.json", "ms_sql_db")
    print(open_ai_dict)
    print(db_dict)

    documents = get_chat_bot_topics_docs(db_dict)

    # parse Document objects into Node objects
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(documents)

    # create vector index to query against
    index = create_vector_index(open_ai_dict, nodes)
    query_engine = index.as_query_engine()
    result = query_engine.query(input)
    response = result.response
    print(f"chat bot response:\n {response}")

    # if the response equals "It is not possible to answer this question with the given context information"
    # store this question/input in a seperate table so that we can provide data to answer it later

    return jsonify(response)


def get_value_from_json(json_file: str, key: str):
    try:
        with open(json_file) as f:
            data = json.load(f)
            return data[key]
    except Exception as e:
        print("Error: ", e)


def get_chat_bot_topics_docs(db_dict: dict) -> list[Document]:
    # connect to database and get all topic_content data from chat_bot_topics table
    topic_content_list = []
    engine = create_engine(
        f"mssql+pyodbc://{db_dict['user']}:{db_dict['password']}@{db_dict['host']}:{db_dict['port']}/Mindset?driver=ODBC+Driver+13+for+SQL+Server"
    )

    with engine.connect() as conn:
        metadata_obj = MetaData()
        metadata_obj.reflect(bind=engine)
        chat_bot_topics = Table("chat_bot_topics", metadata_obj, autoload_with=engine)
        stmt = select(chat_bot_topics).where(chat_bot_topics.c.topic_is_active == 1)
        result = conn.execute(stmt)

        for row in result:
            topic_content_list.append(row.topic_content)

    # create list of documents containing topic data
    documents = [Document(fact) for fact in topic_content_list]
    return documents


def create_vector_index(open_ai_dict: dict, nodes: List[Node]) -> GPTVectorStoreIndex:
    max_input_size = 4096  # set Maximum input size for the LLM.
    tokens_num_output = 256  # set Number of outputs for the LLM.
    max_chunk_overlap = 20  # set Maximum chunk overlap for the LLM.
    chunk_size_limit = 600  # Maximum chunk size to use.

    # define LLM
    llm_predictor = LLMPredictor(
        llm=OpenAI(
            openai_api_key=open_ai_dict["open_ai_key"],
            temperature=0,
            model_name="text-davinci-003",
            max_tokens=tokens_num_output,
        )
    )

    prompt_helper = PromptHelper(
        max_input_size,
        tokens_num_output,
        max_chunk_overlap,
        chunk_size_limit=chunk_size_limit,
    )

    # define service context
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    index = GPTVectorStoreIndex(nodes, service_context=service_context)
    return index


#  main thread of execution to start the server
if __name__ == "__main__":
    app.run()
