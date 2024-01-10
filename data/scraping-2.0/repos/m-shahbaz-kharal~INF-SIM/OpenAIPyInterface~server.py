import socket
import threading
from openai import OpenAI
import time
import json

from utils import receive_data, send_data, log

SLEEP_TIME = 0.5

client = OpenAI()
assistant = client.beta.assistants.create(
    name="Entity",
    instructions="You'll be given an inner thought of any entity; the thought will be self-explanatory, your job is to assist the entity in its queries.",
    tools=[
        {
            "type": "function",
            "function":
            {
                "name": "move",
                "description": "if my intention is to move, I can use this function to move to a specific location.",
                "parameters":
                {
                    "type": "object",
                    "properties":
                    {
                        "position_x": {"type": "number", "description": "x coordinate"},
                        "position_y": {"type": "number", "description": "y coordinate"},
                        "position_z": {"type": "number", "description": "z coordinate"}
                    },
                    "required": ["position_x", "position_y", "position_z"]
                }
            }
        },
        {
            "type": "function",
            "function":
            {
                "name": "affect_self",
                "description": "if my action affects myself, I can use this function.",
                "parameters": {
                    "type": "object",
                    "properties":
                    {
                        "add_to_history": {"type": "string", "description": "the new memories/thoughts to add to my history"},
                        "add_to_health": {"type": "number", "description": "a positive or negative affectation on my health (range: -1.0 to 1.0)"},
                        "movability": {"type": "string", "description": "if I still can move or not after the affectation", "enum": ["True", "False"]},
                    },
                    "required": ["add_to_history", "add_to_health", "movability"]
                }
            } 
        },
        {
            "type": "function",
            "function":
            {
                "name": "affect_other",
                "description": "if my action affects another entity, I can use this function to affect it.",
                "parameters": {
                    "type": "object",
                    "properties":
                    {
                        "entity_id": {"type": "number", "description": "the id of the entity to affect"},
                        "add_to_history": {"type": "string", "description": "the new memories/thoughts to add to the other entity's history"},
                    },
                    "required": ["entity_id", "add_to_history"]
                }
            } 
        },
        {
            "type": "function",
            "function":
            {
                "name": "query_self",
                "description": "if I need to either recall my history, check my health, check my age, or check my location, I can use this function.",
                "parameters": {
                    "type": "object",
                    "properties":
                    {
                        "self_query": {"type": "string", "description": "must be either 'history', 'health', 'age', 'location', 'movability', consciousness', or life_expectancy"},
                    },
                    "required": ["self_query"]
                }
            }
        }
    ],
    model="gpt-3.5-turbo-1106"
)

gather_thoughts_assistant = client.beta.assistants.create(
    name="Thoughts Gatherer",
    instructions="You'll be given inner thoughts of any entity; the thoughts may be redundant or give no extra information, your job is to gather the thoughts and make a single thought out of them.",
    tools=[],
    model="gpt-3.5-turbo-1106"
)

server_socket = socket.socket()
gather_thoughts_socket = socket.socket()
server_socket.bind(('127.0.0.1', 9438))
gather_thoughts_socket.bind(('127.0.0.1', 9439))
server_socket.listen(4)
gather_thoughts_socket.listen(4)

def client_thread_function(client_socket, client_id):
    while True:
        data = receive_data(client_socket)
        log(client_id, f'Received: {data}')
        thread = client.beta.threads.create()
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=data
        )
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )
        while run.status != "completed":
            log(client_id, f'Run status: {run.status}')
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id,
            )
            time.sleep(SLEEP_TIME)
            if run.status in ["queued", "in_progress", "cancelling"]: pass
            elif run.status in ["failed", "cancelled", "expired"]:
                client.beta.threads.delete(thread.id)
                thread = client.beta.threads.create()
                message = client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=data
                )
                run = client.beta.threads.runs.create(
                    thread_id=thread.id,
                    assistant_id=assistant.id
                )
            elif run.status == "requires_action":
                log(client_id, f'Run status: {run.status}')
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []
                for call in tool_calls:
                    log(client_id, f'Tool Call: {call.function.name}({call.function.arguments})')
                    func_args = json.loads(call.function.arguments)
                    func_args_string = ''
                    for func_arg_i, (key, value) in enumerate(func_args.items()):
                        if func_arg_i != 0: func_args_string += '<->'
                        func_args_string += f'{key}:{value}'
                    func_call_string = f'{call.function.name}<->{func_args_string}'
                    send_data(client_socket, func_call_string)
                    func_output = receive_data(client_socket)
                    log(client_id, f'Output: {func_output}')
                    tool_outputs.append(
                        {
                            "tool_call_id": call.id,
                            "output": func_output
                        }
                    )
                run = client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs,
                )
        log(client_id, f'Run status: {run.status}')
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        for message in messages:
            log(client_id, f'Response: {message.content[0].text.value}')
            send_data(client_socket, 'response<->' + message.content[0].text.value)
            break
        client.beta.threads.delete(thread.id)
        log(client_id, '-' * 50)

def client_gather_thoughts_thread_function(client_gather_thoughts_socket, client_id):
    while True:
        data = receive_data(client_gather_thoughts_socket)
        log(client_id, f'Gather Thoughts Request Received: {data}')
        thread = client.beta.threads.create()
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=data
        )
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=gather_thoughts_assistant.id
        )
        while run.status != "completed":
            log(client_id, f'Run status: {run.status}')
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id,
            )
            time.sleep(SLEEP_TIME)
            if run.status in ["queued", "in_progress", "cancelling"]: pass
            elif run.status in ["failed", "cancelled", "expired"]:
                client.beta.threads.delete(thread.id)
                thread = client.beta.threads.create()
                message = client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=data
                )
                run = client.beta.threads.runs.create(
                    thread_id=thread.id,
                    assistant_id=gather_thoughts_assistant.id
                )
        log(client_id, f'Run status: {run.status}')
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        for message in messages:
            log(client_id, f'Gather Thoughts Response: {message.content[0].text.value}')
            send_data(client_gather_thoughts_socket, message.content[0].text.value)
            break
        client.beta.threads.delete(thread.id)
            


client_id = 0
while True:
    log('Server', 'Waiting for connection...')
    client_socket, client_address = server_socket.accept()
    client_gather_thoughts_socket, client_gather_thoughts_address = gather_thoughts_socket.accept()
    log('Server', f'Got connection from {client_address}')
    threading.Thread(target=client_thread_function, args=(client_socket,client_id)).start()
    threading.Thread(target=client_gather_thoughts_thread_function, args=(client_gather_thoughts_socket,client_id)).start()
    client_id += 1