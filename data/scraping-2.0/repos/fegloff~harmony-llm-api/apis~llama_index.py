from flask import request, jsonify
from flask_restx import Namespace, Resource
import openai
import json
import threading

from embeddings import TextArray 
from res import EngMsg as msg
from storages import ChromaStorage

api = Namespace('llama-index', description=msg.API_NAMESPACE_LLMS_DESCRIPTION)

def data_generator(response):
    for chunk in response:
        yield f"data: {json.dumps(chunk)}\n\n"

chromadb = ChromaStorage()
text_array = TextArray(chromadb)
@api.route('/text')
class WebCrawlerTextRes(Resource):
    # 
    # @copy_current_request_context
    def post(self):
        """
        Endpoint to handle LLMs request.
        Receives a message from the user, processes it, and returns a response from the model.
        """ 
        data = request.json
        prompt = data.get('prompt')
        token = data.get('token')
        chatId = data.get('chatId')
        msgId = data.get('msgId')
        url = data.get('url')
        try:
            if prompt and token and chatId and msgId and url:
                thread = threading.Thread(target=text_array.text_query, args=(url, prompt, token, chatId, msgId))
                thread.start()
                return 'OK', 200
            else:
                return "Bad request, parameters missing", 400
        except openai.error.OpenAIError as e:
            error_message = str(e)
            print(f"OpenAI API Error: {error_message}")
            return jsonify({"error": error_message}), 500
        except Exception as e:
            error_message = str(e)
            print(f"Unexpected Error: {error_message}")
            return jsonify({"error": "An unexpected error occurred."}), 500

