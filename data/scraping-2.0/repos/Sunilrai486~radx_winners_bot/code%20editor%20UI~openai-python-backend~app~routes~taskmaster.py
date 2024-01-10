import json

from flask import Blueprint, request

from service.response import response
from service.openai import openai_service

taskmaster_module = Blueprint("taskmaster_module", __name__)

@taskmaster_module.route("/", methods = ["GET"])
def provide_task():
    # The user should be able to choose the course and then 
    # Greet the user and provide a task to the user based on the course. Let's try programming only right now.
    task = openai_service.callopenai("taskmaster")
    body = {
        "response": task,
        "msg": "Task obtained."
    }
    return response(200, body)

@taskmaster_module.route("/query", methods = ["POST"])
def query_about_task():
    # Respond to the query regarding to the task. provide hints only. or maybe this is not required in this challenge. As task master only provides the task.
    data = json.loads(request.data)
    query = data["query"]
    query_response = openai_service.callopenai("taskmaster", query)
    body = {
        "response": query_response,
        "msg": "Query response obtained."
    }
    return response(200, body)