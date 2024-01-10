from flask import Flask, request
from flask_cors import CORS
import os
import openai
import logging
import traceback

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(filename)s <%(funcName)s> %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

log = logging.getLogger(__name__)

openai.api_key = os.getenv('OPENAI_API_KEY')

@app.after_request
def after_request(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    return response

@app.route('/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.get_json(force=True)
        messages = data.get('messages', [])
        log.info('messages: %s', messages)
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=messages,
            temperature=0, # this is the degree of randomness of the model's output
        )
        return {
            "code": 0,
            "data": response
        }
    except Exception as e:
        log.error(e)
        return {
            "code": 1,
            "data": {
                "choices": [
                    {
                        "message": f"error for chat completion {traceback.format_exc()}"
                    }
                ]
            }
        }
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=81)