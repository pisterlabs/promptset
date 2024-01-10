import sys, os
import openai
from langchain.schema import messages_from_dict, messages_to_dict
from flask import Flask, redirect, render_template, request, url_for

import ai_vectorstore
from ai_vectorstore import load_history, save_to_vectorstore
from ai_chains import MyConversation, history_formatter
from ai_roles import ai_init_string

server = Flask(__name__)

AI_ROLE = "default"

# roles are defined in ai_roles.py
# for now there are: "default", "elf", "alien", "divine_ai"

if AI_ROLE not in ai_init_string: AI_ROLE = 'default'

conversation = MyConversation(AI_ROLE, 2000)

CHROMA_ID_FULL_JSON = 'history'

history_dict = load_history(CHROMA_ID_FULL_JSON)
history_text = ''

if history_dict and len(history_dict) > 0:
    history_text = history_formatter(history_dict)
    history_messages = messages_from_dict(history_dict)

    if history_messages:
        conversation.memory.chat_memory.messages = history_messages
        # conversation.memory.predict_new_summary(history_messages, '')

print(f'conversation.memory history={history_text}')


def chat_process_input(human_text):
    global history_dict, history_text
    if human_text == 'clear history':
        history_text = ''
        history_dict = []
        conversation.memory.clear()
        print(f'conversation history cleaned')
        save_to_vectorstore(CHROMA_ID_FULL_JSON, history_dict)
        return 'conversation history cleaned'
    else:
        print(f'sending req. human text={human_text}')
        response_text = conversation.predict(human_input=human_text)
        print(f'response={response_text}')
    history_dict = messages_to_dict(conversation.memory.load_memory_variables({})["history"])
    history_text = history_formatter(history_dict)
    print(f'conversation.memory history={history_text}')
    save_to_vectorstore(CHROMA_ID_FULL_JSON, history_dict)
    return response_text


audio_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recording.webm')


@server.route('/record', methods=['POST'])
def record():
    print(" AUDIO POST REQUEST ")
    file = request.files['audio']
    file.save(audio_file)
    if os.path.exists(audio_file):
        with open(audio_file, 'rb') as file:
            transcript = openai.Audio.transcribe("whisper-1", file, prompt=history_text)
            human_text = transcript['text']
        os.remove(audio_file)
    else:
        human_text = ''
    print('Human text = ', human_text)

    if len(human_text) > 0:
        response_text = chat_process_input(human_text)
    else:
        response_text = ''
    print('POST /record done')
    print(f'Sending response "input": {human_text}, "output": {response_text}')
    return {"input": human_text, "output": response_text}


@server.route('/text_input', methods=['POST'])
def text_input():
    print(" TEXT POST REQUEST ")

    human_text = request.form["text"]

    if len(human_text) > 0:
        response_text = chat_process_input(human_text)
    else:
        response_text = ''
    print('POST /text_input done')
    print(f'Sending response "input": {human_text}, "output": {response_text}')
    return {"input": human_text, "output": response_text}


@server.route("/", methods=("GET", "POST"))
def index():
    def html_formatter(message_list: list[dict]) -> str:
        result = ''
        if len(message_list) > 0:
            for message in message_list:
                result += '<b>' + message["type"].capitalize() + ': </b>' + message["data"]["content"] + "<br>"
        return result

    print(" / GET REQUEST ")

    if history_dict and len(history_dict) > 0:
        result_str = html_formatter(history_dict)
    else:
        result_str = ''
    return render_template("index.html", result=result_str, )


@server.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


def shutdown_server():
    print('SHUTDOWN!')
    ai_vectorstore.vectorstore.persist()
    ai_vectorstore.vectorstore = None
    sys.exit(0)
