import openai
import json
import sys
import time

global  ChatGPTModelToUse
ChatGPTModelToUse = "gpt-3.5-turbo"    # put "gpt-4-0613" to make use of ChatGPT4   

global  ChatGPT4Model
ChatGPT4Model = "gpt-4"

with open('.keys.json', 'r') as f:
    params = json.load(f)
    openai.api_key = params['OPENAI_API_KEY']
    GOOGLE_API_KEY = params['GOOGLE_API_KEY']
    GOOGLE_SEARCH_ENGINE_ID = params['GOOGLE_SEARCH_ENGINE_ID']



global messages

from all_functions import *

def function_needed(myin):
    global messages
    if messages == []:
        messages = ([{"role": "user", "content": myin}])
    else:
        messages.append({"role": "user", "content": myin})

    function_selector_fn = [
            {
                "name": "function_selector",
                "description": """Gets the function_name from the available function list only. Available functions:
                                {'get_weather (gets weather information for a city or location)',
                                'get_current_time',
                                'write_to_file (writes to a file on disk)',
                                'write_python_code_to_file (generates the python code and saves it to disk)',
                                'read_from_file (reads a file from disk)',
                                'show_image (shows and image)',
                                'run_python_code (runs a pyhton code from the saved file)',
                                'search_in_google (used if a question asked that assistant cannot know because the question is about something in present time)',
                                'browse_web_page (browse a web page to get its contents in plain without links and formattings)
                                }
                                """,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_name": {
                            "type": "string",
                            "descriptiopn" : "name of the function to use",
                            "enum": ["get_weather","get_current_time","write_to_file","write_python_code_to_file","read_from_file","show_image","run_python_code","search_in_google","browse_web_page"]},
                    },
                    "required": ["function_name"],
                },
            }

        ]
    try:
        response = openai.ChatCompletion.create(
            model=ChatGPTModelToUse,
            temperature = 0,
            messages=messages,
            functions=function_selector_fn,
            function_call="auto",  # auto is default, but we'll be explicit
        )
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        exception_code = exc_obj.code
        print("Exception code:", exception_code)
        if exception_code == "context_length_exceeded":
            messages= messages[2:]      #deleting first 2 messages from messages
            messages = messages[:-1]    #deleting the last conversation from messages, to prevent duplicate.
            return ("OPENAI_ERROR_TOO_MUCH_CONV")
        else:
            messages = messages[:-1]
            return ('ERROR CONNECTING\n')

    updatetokens(response) 

    i=0
    for resps in response["choices"]:
        i += 1
        if i >1: print (f'Response {i} :  \n{resps["message"]}\n\n')

    response_message = response["choices"][0]["message"]
    fn = response_message.get("function_call")

    if response_message.get("function_call"):
        fn = response_message.get("function_call")
        #funcName=response_message["function_call"]["name"]
        try:
            funcArgs= json.loads(response_message["function_call"]["arguments"])
        except:
            funcArgs = fn["arguments"]
        try:
            nfuncName = funcArgs['function_name']
        except:
            nfuncName = fn['name']

        if nfuncName.lower() =='python': 
            nfuncName = 'run_python_code'
            print(f'👿👿 ERROR! Returned "python" as a command.. trying {nfuncName}')

        print (f'👿 ChatGPT decided to Function to be used: {nfuncName}\n')        
        time.sleep(1)    
        available_functions = {
                    "get_weather": get_weather,
                    "get_current_time" : get_current_time,
                    "write_python_code_to_file" : write_python_code_to_file,
                    "write_to_file" : write_to_file,
                    "read_from_file" : read_from_file,
                    "show_image": show_image,
                    "run_python_code":run_python_code,
                    "search_in_google":search_in_google,
                    "browse_web_page" : browse_web_page,
                }
        if nfuncName not in available_functions:
            return (f'👿👿 ERROR! Tried to get function {nfuncName} - but it is not defined..')
        else:
            function_to_call = available_functions[nfuncName]
            function_interior = get_function_to_use(nfuncName)
            function_response = function_to_call(
                myin=messages,
                function_args=[function_interior],
            )
            messages.append ({"role": "assistant", "content": function_response})
            return(function_response)

    else:
        messages.append ({"role": "assistant", "content": response_message["content"]})
        return (response_message["content"])

def initialize():
    global messages
    cls()
    try:
        with open('messages.json', 'r',encoding='utf-8') as openfile:
            messages = json.load(openfile)
    except:
        init_messages()
    print(f'👻 Loaded {len(messages)} messages from history. Type dump to see messages.')

def clearmemory():
    cls()
    init_messages()
    print('\nConversations Cleared From Memory.\nType SAVE if you want to clear from Disk.')

def init_messages():
    global messages
    dt=get_current_time()
    messages= ([{"role": "user", "content": f'Now is {dt} . Please keep this in mind about when our conversation started.'}])
    messages.append ({"role": "assistant", "content": f'Ok. It is {dt}'})


def handle_exit():
    print ("\nEXITING\n")
    exitSave = ""
    i=0
    while exitSave == "":
        i+=1
        if i >3: 
            exitSave = "N"
        else:
            exitSave = input ("😎 Save Conversation for next chat?  (Y for Yes) :  ")
    if exitSave.lower() in {'y','yes'}:
        with open("messages.json", "w",encoding='utf-8') as outfile:
            json.dump(messages, outfile)
        print('👻 AI   : Conversation Saved and will be automatically loaded in next start')
    else:
        print('👿 AI   : Conversation Discarded')
    exit()

if __name__ == '__main__':
    initialize()
    while True:
        myin = input ("😎 User : ")
        if len(myin)<3 or myin.lower() == "exit":
            handle_exit()
        elif myin.lower() == "cls": 
            cls()
        elif myin.lower() == "save":
            with open("messages.json", "w",encoding='utf-8') as outfile:
                json.dump(messages, outfile)
                print("👻 Saved.")
        elif myin.lower() == "load":
            with open('messages.json', 'r',encoding='utf-8') as openfile:
                messages = json.load(openfile)
                print("👻 Loaded.")
        elif myin.lower() == "new":
            clearmemory()
        elif myin.lower() =="dump":
            print(f'\n===================== DUMP  START =======================')
            try:
                i=0
                for message in messages:
                    i+=1
                    if message['role']=='assistant':
                        print (f"\033[92m👻 {message['role'].ljust(14)}: {message['content']}\033[00m")
                    else:
                        print (f"\033[96m😎 {message['role'].ljust(14)}: {message['content']}\033[00m")
                print(f'\n===================== DUMP FINISH ======================')
                print(f'\nTotal of {i} messages in the memory.')
            except:
                print (messages)
                print(f'\n===================== DUMP FINISH ======================')

        else:
            # If the message is not a direct command like cls / save / load / new / dump then let's ask ChatGPT
            tryno = 1
            while True:
                getresult = function_needed(myin)
                if getresult != "OPENAI_ERROR_TOO_MUCH_CONV":   # if the input is not too much, then the result is retrieved below
                    print(f'👻 AI   : \033[92m{getresult}\033[00m')
                    break
                else:
                    tryno +=1
                    print (f'Reducing the chat size. Trying {tryno}')
                    time.sleep(1)


