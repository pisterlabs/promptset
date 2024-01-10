
import os
import json
import time
from openai import OpenAI


############################################################
#     Inintialize OpenAI client
############################################################
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


############################################################
#     Get or Create the Assistant
############################################################


from meta_assistants.assistants import all_assitants
from meta_assistants.functions import *
from tools import fetch_all_tools

def get_or_create_assistant(assistant_name:str=None):

    assistant_id = None  #TODO: try finding the assistant_id using assistant_name
    if assistant_id:
        assistant = client.beta.assistants.retrieve(assistant_id)

    
    elif assistant_name:    
        assistant_info = all_assitants[assistant_name]
        all_available_tools = fetch_all_tools("meta_assistants/functions.py")
        assitant_tools = [tool for tool in all_available_tools if tool["function"]["name"] in assistant_info["tools"] ]

        assistant = client.beta.assistants.create(
            name= assistant_info["name"],
            instructions=assistant_info["instructions"],
            tools=assitant_tools + [{"type": "retrieval"}],
            model="gpt-4-1106-preview"
        )

    else: 
        return "Must pass either existing assistant_id or an assitant_name to be instantiated"

    return assistant





############################################################
#     Get or Create the Thread
############################################################

def get_or_create_thread(thread_id:str):

    if thread_id:
        thread = client.beta.threads.retrieve(thread_id)

    else:
        thread = client.beta.threads.create()

    print(f"🧵 Thread ID: {thread.id}")
    return thread




############################################################
#     Submit message to the thread
############################################################

def submit_message_to_thread(thread_id:str,message:str, file_ids:list=[]):

    response = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message,
        file_ids=file_ids
    )
    return response



############################################################
#     Run the thread
############################################################

def run_thread(assistant_id, thread_id):
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id
    )

    run = client.beta.threads.runs.retrieve(
        thread_id=thread_id,
        run_id=run.id
    )

    return run


############################
#    Handle function calling:
#    Call local function tools and submit the otuputs back to OpenAI pending thread
############################

def handle_function_calls(thread, run):
    tool_outputs = []
    for tool_call in run.required_action.submit_tool_outputs.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        function_call_output = globals()[function_name](**function_args)
        tool_outputs.append(
            {
                "tool_call_id": tool_call.id,
                "output": function_call_output,
            }
        )

    try:
        run = client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread.id,
            run_id=run.id,
            tool_outputs=tool_outputs
        )
    except Exception as e:
        print(f"too output submission error:{e}")
        print(f'tool output:',tool_outputs)


def listen_for_function_calls(thread, run):
    while True:
        print("🧪 generating:",run.status)
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        if run.status == 'requires_action':
            handle_function_calls(thread, run)
        
        elif run.status == 'in_progress':
            time.sleep(1)
        
        elif run.status == 'completed':
            break

        else:
            time.sleep(2)
    
    return run

############################
#    Handle User Message
############################
def handle_file_uploads(file_paths):
    file_ids = []
    for file_path in file_paths:
        file = client.files.create(
            file=open(file_path, "rb"),
            purpose='assistants'
        )
        file_ids.append(file.id)
    return file_ids


def handle_message(assistant_handle:str, thread_id:str, message:str, file_ids:list=[]):

    # handle file uploads
    # if message.type == "file_upload":
    #     file_ids = handle_file_uploads("./thread_files")

    # get assistant
    assistant = get_or_create_assistant(assistant_name=assistant_handle)

    # get thread
    thread = get_or_create_thread(thread_id=thread_id)

    # submit message to thread 
    submit_message_to_thread(message=message, thread_id=thread.id, file_ids=file_ids)

    # run thread
    run = run_thread(assistant.id, thread.id)
    
    # listen for and handle function calls 
    run = listen_for_function_calls(thread, run)

    
    return {'assistant_handle':assistant_handle, 'thread_id':thread.id, 'final_thread_run_status':run.status }



############################
#    Retrieve Thread Messages
############################

def retrieve_thread_messages(theard_id:str, print_thread=False):

    messages = client.beta.threads.messages.list(
        thread_id=theard_id
    )

    message_list = []
    for message in messages:
        message_dict = {"role": message.role, "content": message.content[0].text.value}
        message_list.append(message_dict)

    if print_thread:
        for message in message_list[::-1]:
            print(f"✉️ 📩 **{message['role']}**: {message['content']}")

    last_completion = next((message['content'] for message in message_list[::-1] if message['role'] == 'assistant'), None),

    return message_list[::-1], last_completion




