import requests
import json
from halo import Halo
from pprint import pprint as pp
import openai
import textwrap
import yaml
import os
from time import sleep
# import SpeechRecognition as sr


###     file operations

def save_yaml(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as file:
        yaml.dump(data, file, allow_unicode=True)



def open_yaml(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data



def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)
    return content
    



def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()


###     create and delete folders and files

def create_folder(folderpath):
    if not os.path.exists(folderpath):
        #mkdirs creates a leaf directory and all intermediate ones, may lead to lots of folders
        os.makedirs(folderpath)
        print("Folder created: ", folderpath)
    else:
        print("Folder already exists: ", folderpath)
        
        
def delete_folder(folderpath):
    if os.path.exists(folderpath):
        os.rmdir(folderpath)
        print("Folder deleted: ", folderpath)
    else:
        print("Folder does not exist: ", folderpath)
        
def create_file(file_name, extension):
    file_path = f"{file_name}.{extension}"
    try:
        with open(file_path, 'w') as file:
            pass  # The file is created and immediately closed
        print(f"File '{file_path}' created successfully.")
    except IOError:
        print(f"An error occurred while creating the file '{file_path}'.")

def delete_file(file_name, extension):
    file_path = f"{file_name}.{extension}"
    try:
        os.remove(file_path)
        print(f"File '{file_path}' deleted successfully.")
    except IOError:
        print(f"An error occurred while deleting the file '{file_path}'.")

        
###     kb functions

def test_create_endpoint(text):
    # text = input("Enter the text for the new KB article: ")
    payload = {"input": text}
    response = requests.post("/create", json=payload) # make sure to get URL hosted
    print('\n\n\n', response.json())
    response = response.json()
    return response



def test_search_endpoint(query):
    # query = input("Enter the search query: ")
    payload = {"query": query}
    response = requests.post("/search", json=payload)
    print('\n\n\n')
    pp(response.json())
    response = response.json()
    return response



def test_update_endpoint(title, text):
    # title = input("Enter the title of the KB article to update: ")
    # text = input("Enter the new text for the KB article: ")
    payload = {"title": title, "input": text}
    response = requests.post("/update", json=payload)
    print('\n\n\n', response.json())
    response = response.json()
    return response



###     tools

def multi_line_input():
    print('\n\n\nType END to save and exit.\n[MULTI] USER:\n')
    lines = []
    while True:
        line = input()
        if line == "END":
            break
        lines.append(line)
    return "\n".join(lines)


functions = [

 {
    "name": "create_folder",
    "description": "Create a folder in the given folderpath if it does not exist",
    "parameters": {
    "type": "object",
    "properties": {
      "folderpath": {
        "type": "string",
        "description": "The folderpath where the folder will be created"
      }
    }
    },
    "required": ["folderpath"]
  },
  {
    "name": "write_file",
    "description": "Create a file with the given filepath and content",
      "parameters": {
      "type": "object",
      "properties": {
        "filepath": {
          "type": "string",
          "description": "The name of the file to write"
        },
        "content": {
          "type": "string",
          "description": "The content of the file to write"
        }
      },
      "required": ["filepath", "content"]
    }
  },
  {
    "name": "delete_folder",
    "description": "Delete a folder in the given folderpath if it exists",
    "parameters": {
    "type": "object",
    "properties": {
      "folderpath": {
        "type": "string",
        "description": "The folderpath where the folder will be deleted"
      }
    }
    },
    "required": ["folderpath"]
  },
  {
    "name": "delete_file",
    "description": "Delete a file with the given file name and extension",
    "parameters": {
    "type": "object",
    "properties": {
      "file_name": {
        "type": "string",
        "description": "The name of the file to be deleted"
      },
      "extension": {
        "type": "string",
        "description": "The extension of the file to be deleted"
      }
    }
    },
    "required": ["file_name", "extension"]
  },
  {
    "name": "test_create_endpoint",
    "description": "Create a new KB article with the given text",
    "parameters": {
    "type": "object",
    "properties": {
      "text": {
        "type": "string",
        "description": "The text for the new KB article"
      }
    }
    },
    "required": ["text"]
  },
  {
    "name": "test_search_endpoint",
    "description": "Search for KB articles that match the given query",
    "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The search query"
      }
    }
    },
    "required": ["query"]
  },
  {
    "name": "test_update_endpoint",
    "description": "Update an existing KB article with the given title and text",
    "parameters": {
    "type": "object",
    "properties": {
      "title": {
        "type": "string",
        "description": "The title of the KB article to update"
      },
      "text": {
        "type": "string",
        "description": "The new text for the KB article"
      }
    }
    },
    "required": ["title", "text"]
  }
]



###    chatbot functions
def chatbot(conversation, model="gpt-3.5-turbo-0613", temperature=0, functions=functions,):
  max_retry = 7
  retry = 0
  while True:
      
            
    # conversation.append({'role': 'user', 'content': user_message})
    system_message = open_file('system_message.txt').replace('<<CODE>>', open_file('scratchpad.txt'))
    conversation.append({'role': 'system', 'content': system_message})
          
    response = openai.ChatCompletion.create(model=model, messages=conversation, temperature=temperature,functions=functions)
            
    bot_message = response['choices'][0]['message']
    # conversation.append({'role': 'assistant', 'content': bot_message})

    if "function_call" in bot_message :
            function_to_call = bot_message["function_call"]
            function_name = function_to_call["name"]
            arguments = json.loads(function_to_call["arguments"])
            function_response = function_call(function_name, arguments)
            # print(function_response)
            conversation.append ({
                    "role": "function",
                    "name": function_name,
                    "content": json.dumps(function_response)
                })
    else:
            user_message = input("\n\nGPT: " + bot_message["content"] + "\n\nYou: ")
            conversation.append({
                      "role": "user",
                      "content": user_message
                    })
      # chatbot(conversation)
 
          


def function_call(function_name, arguments):
   
    if function_name == "test_search_endpoint":
        return test_search_endpoint(**arguments)
    elif function_name == "test_create_endpoint":
        return test_create_endpoint(**arguments)
    elif function_name == "test_update_endpoint":
        return test_update_endpoint(**arguments)
    elif function_name == "write_file":
        return save_file(**arguments)
    elif function_name == "delete_file":
        return delete_file(**arguments)
    elif function_name == "create_folder":
        return create_folder(**arguments)
    elif function_name == "delete_folder":
        return delete_folder(**arguments)
    else:
        print("Function not found")
      
      
def check_scratch(user_message):
  # check if scratchpad updated, continue
    if 'SCRATCHPAD' in user_message:
        user_message = multi_line_input()
        save_file('scratchpad.txt', user_message.strip('END').strip())
        print('\n\n#####      Scratchpad updated!')
            
  # empty submission, probably on accident
    if user_message == '':
        print('You said nothing.')
        return
            

###    main loop

def main():
       # instantiate chatbot
    openai.api_key = open_file('key_openai.txt').strip()
    ALL_MESSAGES = list()
    print('\n\n****** IMPORTANT: ******\n\nType SCRATCHPAD to enter multi line input mode to update scratchpad. Type END to save and exit.')
    while True:
        # # get user input    
        # yn = input('\n\n Speak to the chatbot? (y/n): ')
        
        # if yn == 'y':
        #   listen()
        #   return user_message
        # else:
        user_message = input('\n\n\n[NORMAL] USER:\n\n')
          
        check_scratch(user_message)
        
        # continue with composing conversation and response
        system_message = open_file('system_message.txt').replace('<<CODE>>', open_file('scratchpad.txt'))
        conversation = list()

        conversation.append({'role': 'user', 'content': user_message})
        conversation.append({'role': 'system', 'content': system_message})
        # generate a response
        # spinner = Halo(text='Coding...', spinner='dots')
        # spinner.start()
        chatbot(conversation)
        # spinner.stop()
        
        

        
        




if __name__ == "__main__":
    main()
    
    
    
    
        # # check if scratchpad updated, continue
        # if 'SCRATCHPAD' in text:
        #     text = multi_line_input()
        #     save_file('scratchpad.txt', text.strip('END').strip())
        #     print('\n\n#####      Scratchpad updated!')
        #     continue
        # if text == '':
        #     # empty submission, probably on accident
        #     continue
              
            # if "function_call" in response:
            #   function_response = function_call(response)
            #   conversation.append({
            #           "role": "function",
            #           "name": response["choices"][0]["message"]["function_call"]["name"],
            #           "content": json.dumps(function_response)
            #       })
              
            #   return function_response
    
        # print('\n\n\n\nCHATBOT:\n')
        # print(response)
        
        
        
        
        
        # while response["choices"][0]["finish_reason"] == "function_call":
        #     function_response = function_call(response)
        #     conversation.append({
        #         "role": "function",
        #         "name": response["choices"][0]["message"]["function_call"]["name"],
        #         "content": json.dumps(function_response)
        #     })

        #     print("Function Call Response: ", conversation[-1]) 
    
    
    
    
    #  print("\n\n\n1. Create KB article")
    #     print("2. Search KB articles")
    #     print("3. Update KB article")
    #     print("4. Exit")
    #     choice = input("\n\nEnter your choice: ")
    #     if choice == '1':
    #         test_create_endpoint()
    #     elif choice == '2':
    #         test_search_endpoint()
    #     elif choice == '3':
    #         test_update_endpoint()
    #     elif choice == '4':
    #         break
    #     else:
            
    #         print("\n\n\nInvalid choice. Please enter a number between 1 and 4.")



# def ask_function_calling(conversation, functions, tokens):
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo-16k-0613",
#         messages=conversation,
#         functions = functions,
#         function_call="auto"
#     )
#     if tokens > 15000:
#         conversation.pop(0)
#         conversation.append({'role': 'assistant', 'content': response})
        
#     print(response)

#     function_response = function_call(response)
#     conversation.append({
#             "role": "function",
#             "name": response["choices"][0]["message"]["function_call"]["name"],
#             "content": json.dumps(function_response)
#         })

#     print("messages: ", conversation) 

#     response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo-16k-0613",
#             messages=conversation,
#             functions = functions,
#             function_call="auto"
#         )   

#     print("response: ", response) 
    
# def print_formatted(response):
#       formatted_lines = [textwrap.fill(line, width=120) for line in response.split('\n')]
#       formatted_text = '\n'.join(formatted_lines)
#       print(formatted_text)
           
        # except Exception as oops:
        #     print(f'\n\nError communicating with OpenAI: "{oops}"')
        #     if 'maximum context length' in str(oops):
        #         a = conversation.pop(0)
        #         print('\n\n DEBUG: Trimming oldest message')
        #         continue
        #     retry += 1
        #     if retry >= max_retry:
        #         print(f"\n\nExiting due to excessive errors in API: {oops}")
        #         exit(1)
        #     print(f'\n\nRetrying in {2 ** (retry - 1) * 5} seconds...')
        #     sleep(2 ** (retry - 1) * 5)
       
        
        # funcs = response.to_dict()['function_call']
        # funcs = json.loads(funcs)
        

        # while funcs != None:
        #   if tokens > 15000:
        #     conversation.pop(0)
        #     conversation.append({'role': 'assistant', 'content': response})
            
        #     function_response = function_call(response)
        #     conversation.append({
        #         "role": "function",
        #         "name": funcs["name"],
        #         "content": json.dumps(function_response)
        #     })

        #     print("messages: ", conversation[-1]) 

        #     response = openai.ChatCompletion.create(
        #         model="gpt-3.5-turbo-16k-0613",
        #         messages=conversation,
        #         functions = functions,
        #         function_call="auto"
        #     )   
            
        #     conversation.append({"role": "assistant", "content": response})

        #     print("response: ", response) 
        # else:
        #     print(response)