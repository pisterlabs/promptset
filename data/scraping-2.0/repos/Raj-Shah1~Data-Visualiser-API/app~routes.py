from flask import Response, request, jsonify
from app import app
from app.openai_api import openai_query_generation
from app.db import create_sql_queries_table, create_sql_queries_record, get_sql_queries_by_name
from app.db import get_sql_queries, update_sql_query_record, delete_sql_query_record, get_sql_queries_by_id
from app.service.fetch_graph_data import table_data
import json


@app.route("/queries/generate", methods=["POST"])
def generate_query_route():

    json_data = request.json
    if json_data is None or 'question' not in json_data:
        return "Invalid request data", 400

    question = str(json_data['question'])
    if question:
        answer = openai_query_generation(question)
        return Response(answer, mimetype="text/plain")
    else:
        return "Invalid or empty question. Please provide a valid question.", 400  


@app.route("/queries/execute", methods=["POST"])
def execute_query_route():
    
    json_data = request.json
    if json_data is None or 'query' not in json_data:
        return "Invalid request data", 400
    
    query = str(json_data['query'])

    google_table_data = table_data(query)

    if google_table_data:
        # Convert the data to JSON and encode it as bytes
        json_data_bytes = json.dumps(google_table_data).encode('utf-8')
        return Response(json_data_bytes, mimetype="application/json")
    else:
        return "Invalid or empty query. Please provide a valid query.", 400


@app.route("/queries", methods=["POST"])
def save_query_route():
    json_data = request.json
    if json_data is None:
        return "Invalid request data", 400
    
    query = str(json_data['query'])
    name = str(json_data['name'])

    if name is None or name == "":
        return "Invalid or empty query name. Please provide a valid query name.", 400
    
    if query is None or query == "":
        return "Invalid or empty query. Please provide a valid query.", 400

    create_sql_queries_table()
    sql_query_data = get_sql_queries_by_name(name)

    if not sql_query_data:  # Check if the query does not already exist
        create_sql_queries_record(name, query)
        return "Query saved successfully.", 201
    else:
        return "Record with the same query name already exists. Kindly update the query name", 409
    

@app.route("/queries", methods=["GET"])
def get_query_route():
    query_result = get_sql_queries()
    json_data_bytes = json.dumps(query_result).encode('utf-8')
    return Response(json_data_bytes, mimetype="application/json")


@app.route("/queries/<int:id>", methods=["PUT"])
def update_query_route(id):
    json_data = request.json
    if json_data is None or 'query' not in json_data:
        return "Invalid request data", 400
    
    query = str(json_data['query'])

    if query is None or query == "":
        return "Invalid or empty query. Please provide a valid query.", 400

    sql_query_data_by_id = get_sql_queries_by_id(id)

    if sql_query_data_by_id is None:
        return "Query Not Found", 404
    
    update_sql_query_record(id, query)
    return "Query saved successfully.", 200
    

@app.route("/queries/<int:id>", methods=["DELETE"])
def delete_query_route(id):
    print("Request received to delete query with id: ", id)
    
    sql_query_data_by_id = get_sql_queries_by_id(id)

    if sql_query_data_by_id is None:
        return "Query Not Found", 404
    
    delete_sql_query_record(id)
    return "Query deleted successfully.", 204