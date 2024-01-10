from flask import Blueprint, request, jsonify
import openai
from langchain.chat_models import ChatOpenAI
from .bot.base import create_agent
from os import environ

print("STARTING UP API ROUTES")

bp = Blueprint('routes', __name__)

openai.api_key = environ["OPENAI_API_KEY"]

@bp.route('/', methods=['OPTIONS'])
def handle_options():
    return '', 204

@bp.route('/chat', methods=['POST'])
def test():    
    data = request.get_json()
    
    print(data)
    text = data.get('text', '')
    
    print("GOT:", text)
    
    query_dict = {
        'input': text,
    }

    agent = create_agent(
        llm=ChatOpenAI(model="gpt-4-1106-preview", temperature=0), 
        verbose=True, 
        handle_parsing_errors=True,  
        query=query_dict
    )
    
    result = agent.run(query_dict)
    return jsonify(result)
