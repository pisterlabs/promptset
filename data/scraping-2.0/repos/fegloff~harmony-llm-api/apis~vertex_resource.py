from flask import request, jsonify
from flask_restx import Namespace, Resource
from vertexai.language_models import ChatModel, ChatMessage
from google.oauth2 import service_account
import google.cloud.aiplatform as aiplatform
from litellm import litellm
import openai
import vertexai
import json

from res import EngMsg as msg

with open(
    "res/service_account.json"
) as f:
    service_account_info = json.load(f)

my_credentials = service_account.Credentials.from_service_account_info(
    service_account_info
)

aiplatform.init(
    credentials=my_credentials,
)

with open("res/service_account.json", encoding="utf-8") as f:
    project_json = json.load(f)
    project_id = project_json["project_id"]


litellm.vertex_project = project_id
litellm.vertex_location = "us-central1"
vertexai.init(project=project_id, location="us-central1")

api = Namespace('vertex', description=msg.API_NAMESPACE_VERTEX_DESCRIPTION)

@api.route('/completions') 
class VertexCompletionRes(Resource):
    
    def post(self): 
        """
        Endpoint to handle Google's Vertex/Palm2 LLMs.
        Receives a message from the user, processes it, and returns a response from the model.
        """
        data = request.json
        if data.get('stream') == "True":
            data['stream'] = True # convert to boolean

        try:
            if data.get('stream') == "True":
                data['stream'] = True # convert to boolean
            # pass in data to completion function, unpack data
            
            chat_model = ChatModel.from_pretrained("chat-bison@001")
            parameters = {
                "max_output_tokens": 800,
                "temperature": 0.2
            }
            prompt = data.get('messages')[-1]
            messages = data.get('messages')
            messages.pop()
            history = [ChatMessage(item.get('content'), item.get('author')) for item in messages]
            chat = chat_model.start_chat(
                max_output_tokens=800,
                message_history=history
            )
            response = chat.send_message(f"{prompt.get('content')}", **parameters)
            # if data['stream'] == True: # use generate_responses to stream responses
            #     return Response(data_generator(response), mimetype='text/event-stream')
            
            return f"{response}", 200 # non streaming responses
        except openai.error.OpenAIError as e:
            # Handle OpenAI API errors
            error_message = str(e)
            print(f"OpenAI API Error: {error_message}")
            return jsonify({"error": error_message}), 500
        except Exception as e:
            # Handle other unexpected errors
            error_message = str(e)
            print(f"Unexpected Error: {error_message}")
            return jsonify({"error": "An unexpected error occurred."}), 500
