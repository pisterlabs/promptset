import sys
# app.py
from flask import Flask, request, render_template
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_login import current_user, login_user, logout_user, login_required
import autogen, re
from flask_cors import CORS
from groupchat_flask import groupchat_a
import base64
import os
from openai import OpenAI
import mimetypes
app = Flask(__name__)
CORS(app, supports_credentials=True)


#------------------add temp dir
temp_dir = './tmp/'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
    
# ---------------- config

model_type=[{'model_type': 'GPT'}]

gpt=[{'model':'gpt-3.5-turbo',
      'api_key':'',}]

llm=[{'model':'mistral-7b',
        'api_key':'',
        'base_url':'',}]

# print('Model type before condition:', model_type[0]['model_type'])

# if model_type[0]['model_type'] == 'GPT':
#     # GPT

#     config_list = [{
#         "model": gpt[0]['model'],
#         "api_key":gpt[0]['api_key'],
#         #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
#     }]

#     print('GPT is running, model name:', gpt[0]['model'], 'API key:', gpt[0]['api_key'])

#     # LLM
# elif model_type[0]['model_type'] == 'LLM':
#     config_list = [{
#         "model": llm[0]['model'],
#         "api_key": "NULL",
#         "base_url": llm[0]['base_url'],
#     }]

#     print('LLM is running, model name:', llm[0]['model'],  'base url:', llm[0]['base_url'])

# print('Model type before condition:', model_type[0]['model_type'])


# -----------------config_list_gpt4


# -----------------socketio
socketio = SocketIO(app, cors_allowed_origins='*')
@app.route('/')
def index():
    return render_template('frontend_runchat.html')

pattern = re.compile(r"You: (.+?)\nAgent: (.+?)(?=You:|$)", re.DOTALL)

user_proxys = {}
assistants = {}
file_dict = {}

# ------model type------
# SocketIO event for disconnecting
@socketio.on('update_model_type')
def handle_update_model_type(message):
    new_model_type = message['model_type']
    #update model type
    model_type[0]['model_type'] = new_model_type
    print('Model Type updated:', new_model_type)

# ------GPT------
# SocketIO event for disconnecting
@socketio.on('update_GPTmodel_name')
def handle_update_model_name(message):
    new_model_name = message['GPTmodel_name']
    # Here we update model name
    gpt[0]['model'] = new_model_name
    print('GPTModel Name updated:', new_model_name)

# SocketIO event for updating API key
@socketio.on('update_api_key')
def handle_update_api_key(message):
    new_api_key = message['api_key']
    # Here we update the first configuration's API key
    gpt[0]['api_key'] = new_api_key
    print('API Key updated:', new_api_key)

# ------LLM------
# SocketIO event for disconnecting
@socketio.on('update_LLMmodel_name')
def handle_update_model_name(message):
    new_model_name = message['LLMmodel_name']
    # Here we update the first configuration's API key
    llm[0]['model'] = new_model_name
    print('LLMModel Name updated:', new_model_name)


# SocketIO event for updating base url
@socketio.on('update_LLM_base_url')
def handle_update_base_url(message):
    new_base_url = message['base_url']
    # Here we update the first configuration's API key
    llm[0]['base_url'] = new_base_url
    print('Base URL updated:', new_base_url)

config_list = [{
        "model": gpt[0]['model'],
        "api_key":gpt[0]['api_key'],
        #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
    }]

config_list_gpt4 = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0,
    "config_list": config_list,
    "timeout": 120,
}
def update_config_list():
    if model_type[0]['model_type'] == 'GPT':
        config_list_gpt4['config_list'] = [{
            "model": gpt[0]['model'],
            "api_key": gpt[0]['api_key']
        }]
    elif model_type[0]['model_type'] == 'LLM':
        config_list_gpt4['config_list'] = [{
            "model": llm[0]['model'],
            "api_key": "NULL",
            "base_url": llm[0]['base_url']
        }]
    print('config_list', config_list_gpt4['config_list'])

# ----upload file----
# SocketIO event for uploading a file
@socketio.on('file-upload')
def handle_file_upload(json):
    print('receive file from', request.sid)
    print('receive file:' + json['name'])
    file_data = base64.b64decode(json['data'])
    file_name = json['name']
    save_path = os.path.join(temp_dir, file_name)  # Specify your directory path
    file_dict[request.sid] = save_path
    with open(save_path, 'wb') as file:
        file.write(file_data)
    user_proxys[request.sid], assistants[request.sid] = groupchat_a(config_list_gpt4,request.sid,doc_path=file_dict[request.sid])
    print('agent created')
    return 'File uploaded successfully'

# SocketIO event for downloading a file
@socketio.on('connect')
def handle_connect():
    update_config_list()
    join_room(request.sid)
    user_proxys[request.sid], assistants[request.sid] = groupchat_a(config_list_gpt4,request.sid)
    file_path = './flask_React.py'

    # with open(file_path, 'rb') as file:
    #     file_data = file.read()
    #     encoded_data = base64.b64encode(file_data).decode('utf-8')  # Encode to base64 and then decode to string

    #     # Optionally, get the MIME type of the file
    #     mime_type, _ = mimetypes.guess_type(file_path)
    #     if not mime_type:
    #         mime_type = 'application/octet-stream'  # Default MIME type if unknown

    #     # Create a dictionary to hold the file data and additional info
    #     file_info = {
    #         'fileData': encoded_data,
    #         'fileName': file_path.split('/')[-1],  # Extract the file name
    #         'mimeType': mime_type
    #     }

    #     emit('file_received', file_info, room=request.sid)
    print('connect', request.sid)


# SocketIO event for chat response
@socketio.on('message')
def handle_message(message):
    print('receive message', message)
    user_input = message.get('content')
    user_proxy = user_proxys[request.sid]
    # ----xin add
    if user_proxy.name == "Content_Assistant":
        user_proxy.initiate_chat(assistants[request.sid], problem=user_input,clear_history=False)
    else:
        user_proxy.initiate_chat(assistants[request.sid], message=user_input, clear_history=False)
        # ----xin add
          # save dialogue for differenet user in dialogue
        # matches = pattern.findall(dialogue)
        # dialogue = [{"you": match[0], "agent": match[1]} for match in matches]


if __name__ == '__main__':
    socketio.run(app, debug=False,port=5003)
