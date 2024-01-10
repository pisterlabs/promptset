from flask import request, jsonify, Response, make_response, current_app as app
from flask_restx import Namespace, Resource
from litellm import completion
from openai.error import OpenAIError
import openai
import json

from res import EngMsg as msg

api = Namespace('llms', description=msg.API_NAMESPACE_LLMS_DESCRIPTION)

def data_generator(response):
    for chunk in response:
        yield f"data: {json.dumps(chunk)}\n\n"

@api.route('/completions') 
class LlmsCompletionRes(Resource):
    
    def post(self):
        """
        Endpoint to handle LLMs request.
        Receives a message from the user, processes it, and returns a response from the model.
        """ 
        app.logger.info('handling llm request')
        data = request.json
        if data.get('stream') == "True":
            data['stream'] = True # convert to boolean
        # if not data:

        #     return jsonify({"error": "Invalid request data"}), 400
        try:
            if data.get('stream') == "True":
                data['stream'] = True # convert to boolean
            # pass in data to completion function, unpack data
            response = completion(**data)
            if data['stream'] == True: 
                return Response(data_generator(response), mimetype='text/event-stream')
        except OpenAIError as e:
            # Handle OpenAI API errors
            error_message = str(e)
            app.logger.error(f"OpenAI API Error: {error_message}")
            return jsonify({"error": error_message}), 500
        except Exception as e:
            # Handle other unexpected errors
            error_message = str(e)
            app.logger.error(f"Unexpected Error: {error_message}")
            return jsonify({"error": "An unexpected error occurred."}), 500
        # return response, 200
        return make_response(jsonify(response), 200)

