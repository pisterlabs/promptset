import os, sys, re
from modules.functions import *
from modules.helpers import *
from print_color import print
from openai import OpenAI
from os import getenv

# gets API Key from environment variable OPENAI_API_KEY
client = OpenAI(base_url="https://openrouter.ai/api/v1",api_key=getenv("OPENROUTER_API_KEY"),)

def response_generator(message_dict):
    completion = client.chat.completions.create(model="nousresearch/nous-capybara-34b", messages=message_dict)
    response = completion.choices[0].message.content
    return response
    
#LLM Selection + History init
llm_name = char_selector()

history_path = os.path.join("history", f'history_{llm_name}.csv')
try:
    with open(history_path, 'r', newline='') as file:
        reader = csv.DictReader(file)
        history_dict = list(reader)
        for key in history_dict:
            history = ''
            history = history + prompter(key, history_dict[key])
except FileNotFoundError:
    history = ''
    history_dict = []

#Chat
while True:
    role_select = input("Select role - system/user :> ")
    if role_select == 'system':
        system_message = input('System:> ')
        history, history_dict = history_update_print('system', history, history_dict, system_message)
    
    user_message = input("User:> ")
    if user_message == 'exit':
        sys.exit(0)  
    history, history_dict = history_update_print('user', history, history_dict, user_message)

    assistant_message = response_generator(history)
    history, history_dict = history_update_print('assistant', history, history_dict, assistant_message, True, llm_name)
    print(assistant_message, tag=llm_name, tag_color='magenta', color='cyan')
        
    if '/function' in assistant_message:
        matches = re.findall(r'\[([^]]+)\]', assistant_message)
        extracted_info = {}    
        function_name = matches[0]
        try:
            args_str = matches[1]
            if function_name in globals() and callable(globals()[function_name]):
                func_to_call = globals()[function_name]  
                args = [arg.strip() for arg in args_str.split(',')]
                response = func_to_call(*args)
            else:
                response = 'Function not found'
        except:
            if function_name in globals() and callable(globals()[function_name]):
                func_to_call = globals()[function_name]
                response = func_to_call()
            else:
                response = 'Function not found'                
        history, history_dict = history_update_print('system', history, history_dict, system_message)
        print(response, tag='System', tag_color='yellow', color='white')

        assistant_message = response_generator(history)        
        history, history_dict = history_update_print('assistant', history, history_dict, assistant_message, True, llm_name)
        print(assistant_message, tag=llm_name, tag_color='magenta', color='cyan')