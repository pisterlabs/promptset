from flask import Blueprint, render_template, request

from app.models import Result
from app.services.open_ai import OpenAIService

routes = Blueprint('routes', __name__)


@routes.route('/')
def home():
    """Render response from OpenAI API query"""
   
    query = request.args.get('query', None)

    if query is None:
        return render_template('response_view.html')

    openai_service = OpenAIService()
    response = openai_service.query(query)

    data_list = []
    query_message = Result(time="This Time", messagetype="other-message float-right", message=query)
    response_message = Result(time="This Time", messagetype="my-message", message=response)

    data_list.append(query_message)
    data_list.append(response_message)
    return render_template('response_view.html', results=data_list)
