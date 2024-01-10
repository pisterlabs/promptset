import json
import time
import queue
from flask import Flask, request, Response, render_template, jsonify, redirect, url_for, stream_with_context
from flask_cors import CORS, cross_origin
from typing import Any, List
from pydantic import BaseModel
from datetime import datetime
from dotenv import load_dotenv
import threading
from llama2_utils import llama2_prompt
from model_utils import create_llama2, create_llama2_v2
from dataclasses import dataclass
from langchain.callbacks.base import BaseCallbackHandler
from llm_utils import ChatMessage, LLmTaskItem, LlmCallbackHandler
load_dotenv()


llm_queue = queue.Queue(10)
llm_callback_handler = LlmCallbackHandler()
print(f"loading llm")
llm = create_llama2_v2(llm_callback_handler)


class LlmConsumer(threading.Thread):
    def __init__(self, thread_name):
        super(LlmConsumer, self).__init__(name=thread_name)

    def run(self):
        global llm_queue
        while True:
            if llm_queue.empty():
                time.sleep(1)
                continue
            task_item: LLmTaskItem = llm_queue.get()
            llm_callback_handler.current_task_item = task_item
            print(f"start process")
            resp = llm(task_item.messages)
            task_item.output_message = resp
            task_item.wait_for_response_done()


llm_task = LlmConsumer('consumer')
llm_task.daemon = True
llm_task.start()
print(f"consumer task started")

app = Flask(__name__)
cors = CORS(app)


@app.route('/api/v1/chat/completions', methods=['POST'])
def chat_completions():
    req = request.json
    messages = req['messages']
    resp = llm(llama2_prompt(messages))
    result = {
        'outputs': resp,
    }
    return jsonify(result)


@app.route('/api/v1/chat/stream', methods=['POST'])
def chat_stream():
    req = request.json
    messages = req['messages']
    task_item = LLmTaskItem()
    task_item.messages = llama2_prompt(messages)
    llm_queue.put(task_item)
    task_item.wait_for_start()
    return Response(stream_with_context(task_item.response()), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True, threaded=True, use_reloader=False)
