import os
import ast
import openai

from dotenv import load_dotenv
from terminal import terminal_access
from di_code import dynamic_code_execution
from complexity import complexity_analyzer
from profiler import code_profiler


# Load environment variables from .env file
load_dotenv()

# Access the API key from the environment variable
openai_api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = openai_api_key


def write_to_file(content, file_path):
    """
    Writes the given content to the file at the given file path.

    Args:
        content (str): The content to write to the file.
        file_path (str): The path of the file to write to, relative to /home/stahlubuntu/coder_agent/bd/.
    """
    if not os.path.exists('/home/stahlubuntu/coder_agent/bd/' + file_path):
        open('/home/stahlubuntu/coder_agent/bd/' + file_path, 'w').close()
    with open('/home/stahlubuntu/coder_agent/bd/' + file_path, 'w') as f:
        f.write(content)

def read_file(file_path):
    """
    Reads the contents of the file at the given file path.

    Args:
        file_path (str): The path of the file to read from, relative to /home/stahlubuntu/coder_agent/bd/.

    Returns:
        str: The contents of the file.
    """
    with open('/home/stahlubuntu/coder_agent/bd/' + file_path, 'r') as f:
        return f.read()

def pretty_print_conversation(messages):
    """
    Prints the conversation in a pretty format.

    Args:
        messages (list): A list of messages in the conversation.
    """
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }
    for message in messages:
        color = role_to_color.get(message["role"], "white")
        if message["role"] == "function":
            print(colored(f'{message["role"]}: {message["name"]} output: {message["content"]}', color))
        else:
            print(colored(f'{message["role"]}: {message["content"]}', color))

def search_code_completion_request(messages, functions):
    """
    Makes a request to the OpenAI Chat Completions API for code search.

    Args:
        messages (list): A list of messages in the conversation.
        functions (dict): A dictionary of functions to be used in the conversation.

    Returns:
        dict: The response from the API.
    """
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=messages,
        functions=functions,
        function_call={"auto"}
    )




def print_numerated_history(conversation):
    """Prints the chat history with each message prefixed by its number.
    
    Args:
        conversation (list): The list of messages in the conversation.
    """
    for idx, message in enumerate(conversation, 1):
        print(f"{idx}. {message['role'].capitalize()}: {message['content']}")

def remove_messages_by_indices(conversation, indices):
    """Removes messages from the conversation based on the provided indices.
    
    Args:
        conversation (list): The list of messages in the conversation.
        indices (list): The indices of the messages to remove.
        
    Returns:
        list: The updated conversation with messages removed.
    """
    for index in sorted(indices, reverse=True):
        if 0 < index <= len(conversation):
            del conversation[index - 1]
        else:
            print(f"Invalid index: {index}")
    return conversation






def ast_tool(code: str, analyze_functions: bool = False, analyze_variables: bool = False, analyze_control_flow: bool = False) -> str:
    """
    Analyzes the Abstract Syntax Tree (AST) of a given Python code.
    
    Parameters:
    - code (str): The Python code that needs to be analyzed.
    - analyze_functions (bool): Whether to analyze function calls in the code.
    - analyze_variables (bool): Whether to analyze variable assignments in the code.
    - analyze_control_flow (bool): Whether to analyze control flow structures like loops and conditionals.
    
    Returns:
    - str: A string representation of a dictionary containing the analysis results.
    """
    
    # Parse the code into an AST
    tree = ast.parse(code)
    
    # Initialize results dictionary
    results = {
        'functions': [],
        'variables': [],
        'control_flow': []
    }
    
    # Walk through the AST nodes
    for node in ast.walk(tree):
        if analyze_functions and isinstance(node, ast.Call):
            results['functions'].append(node.func.id if hasattr(node.func, 'id') else str(node.func))
        if analyze_variables and isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    results['variables'].append(target.id)
        if analyze_control_flow:
            if isinstance(node, ast.If):
                results['control_flow'].append('if_statement')
            elif isinstance(node, ast.For):
                results['control_flow'].append('for_loop')
            elif isinstance(node, ast.While):
                results['control_flow'].append('while_loop')
    
    # Convert the results dictionary to a string
    results_str = json.dumps(results, indent=4)
    
    return results_str

import pdb
import json




import pdb
import io
from contextlib import redirect_stdout

def pdb_tool(code_or_filename: str, debug_operations: dict) -> str:
    # Initialize the result messages list
    result_messages = []
    
    # Folder path for code files
    folder_path = "/home/stahlubuntu/coder_agent/bd/"
    
    # Validate code_or_filename
    if not isinstance(code_or_filename, str):
        raise ValueError("code_or_filename must be a string")
    
    # Check if code input is a filename
    if code_or_filename.endswith('.py'):
        with open(folder_path + code_or_filename, 'r') as file:
            code = file.read()
        result_messages.append(f"Reading file {code_or_filename}")
    else:
        code = code_or_filename
        result_messages.append(f"Debugging code snippet")
    
    result_messages.append(f"Debugging target: {code_or_filename}")

    # Prepare PDB commands
    pdb_commands = []

    # Validate and set breakpoints
    if "set_breakpoints" in debug_operations:
        for bp in debug_operations["set_breakpoints"]:
            if not isinstance(bp, int):
                raise ValueError("Each breakpoint must be an integer")
            pdb_commands.append(f"b {bp}")
            result_messages.append(f"Breakpoint set at line {bp}")

    # Validate and step through code
    if "step" in debug_operations and debug_operations["step"]:
        if not isinstance(debug_operations["step"], bool):
            raise ValueError("Step operation must be a boolean")
        pdb_commands.append("s")
        result_messages.append("Stepping through code")

    # Validate and continue execution
    if "continue" in debug_operations and debug_operations["continue"]:
        if not isinstance(debug_operations["continue"], bool):
            raise ValueError("Continue operation must be a boolean")
        pdb_commands.append("c")
        result_messages.append("Continuing execution")

    # Validate and inspect variables
    if "inspect" in debug_operations:
        for var in debug_operations["inspect"]:
            if not isinstance(var, str):
                raise ValueError("Each variable to inspect must be a string")
            pdb_commands.append(f"p {var}")
            result_messages.append(f"Inspecting variable {var}")

    # Run the PDB session with prepared commands
    with io.StringIO() as buffer, redirect_stdout(buffer):
        debugger = pdb.Pdb(stdin=io.StringIO("\n".join(pdb_commands)), stdout=buffer)
        debugger.run(code)

        # Append any PDB output
        pdb_output = buffer.getvalue()
        if pdb_output:
            result_messages.append(f"PDB Output:\n{pdb_output}")

    return "\n".join(result_messages)




# Changing the function name to `automated_code_reviewer` as requested, while keeping the string output format.

import subprocess


def automated_code_reviewer(code_or_filename):
    """
    Perform an automated code review using Pylint and return all categories as a string.

    Parameters:
        code_or_filename (str): The Python code or filename to review.

    Returns:
        str: String-formatted review feedback.
    """
    
    # Initialize the review_feedback dictionary
    review_feedback = {}
    
    # Constant folder path for code files
    FOLDER_PATH = "/home/stahlubuntu/coder_agent/bd/"

    try:
        # Check if code input is a filename
        if code_or_filename.endswith('.py'):
            with open(f"{FOLDER_PATH}{code_or_filename}", 'r') as file:
                code = file.read()
        else:
            code = code_or_filename

        # Save code to a temporary file to run pylint
        with open("temp_code.py", "w") as temp_file:
            temp_file.write(code)

        # Run pylint on the code and capture its output
        pylint_output = subprocess.getoutput(f"pylint temp_code.py")

        return pylint_output
    
    except FileNotFoundError:
        return "Error: File not found."
    except Exception as e:
        return f"An error occurred: {e}"






# %%
import json
import openai
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
import os
import json
import requests
from termcolor import colored
from dotenv import load_dotenv
from code_search import similarity_search
import openai
from testt import unit_test_runner
from functions_ca import functions3
import json
import openai
from code_search import similarity_search
from metaphor import metaphor_web_search
from code_reviwer import automated_code_reviewer
from scrape import scrape_web_pages


# Define the GPT models to be used
GPT_MODEL1 = "gpt-3.5-turbo-0613"
GPT_MODEL = "gpt-4-0613"

# Load environment variables from .env file
load_dotenv()

# Access the API key from the environment variable
openai_api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = openai_api_key

def pretty_print_conversation(messages):
    """
Prints a conversation between a system, user, and assistant in a readable format.

The conversation is passed in as a list of message dictionaries. Each message has a
"role" (system, user, assistant) and "content". 

This function loops through the messages and prints them with the role name and  
content. The text is color coded based on the role using the role_to_color mapping.

Special handling is done for messages with the "function" role to print the function
name and output.

Parameters:
    messages (list): The list of message dictionaries representing the conversation.
        Each message dict contains "role" and "content" keys.
        
Returns:
    None
"""
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }
    for message in messages:
        color = role_to_color.get(message["role"], "white")
        if message["role"] == "function":
            print(colored(f'{message["role"]}: {message["name"]} output: {message["content"]}', color))
        else:
            print(colored(f'{message["role"]}: {message["content"]}', color))





import openai
import tiktoken
import json

# Get the CL100KBase encoding and create a Tokenizer instance
cl100k_base = tiktoken.get_encoding("cl100k_base")
tokenizer = cl100k_base

max_response_tokens = 250
token_limit = 7000  # Adjusted to your value
#conversation = []
#conversation.append(system_message)

def num_tokens_from_messages(messages):
    """
Calculates the number of tokens used so far in a conversation.

This uses the CL100KBase tokenizer to encode the conversation messages
and tally up the total number of tokens.

Each message is encoded as:
<im_start>{role/name}\n{content}<im_end>\n

Parameters:
    messages (list): The list of conversation messages. 
        Each message is a dict with "role" and "content" keys.
        
Returns:
    num_tokens (int): The total number of tokens used in the conversation.
"""
    encoding = tiktoken.get_encoding("cl100k_base")  # Model to encoding mapping
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            if isinstance(value, str):  # Ensure value is a string before encoding
                num_tokens += len(encoding.encode(value))
                if key == "name":  # If there's a name, the role is omitted
                    num_tokens -= 1  # Role is always required and always 1 token
    num_tokens += 2  # Every reply is primed with <im_start>assistant
    return num_tokens




"""
This code defines a conversational agent that can analyze and improve code snippets.

It initializes the conversation with an introductory system message and a user
message containing the code snippets to analyze. 

It then enters a loop to collect user input messages, check for slash commands,
and pass the conversation to the OpenAI Chat Completions API if it is under the
token limit.

The num_tokens_from_messages() function calculates the number of tokens used so far.

If the API response contains an assistant message, it is printed and added to the
conversation. If it contains a function call, that function is executed and the 
output is printed.

The similarity_search() function can be called to fetch relevant code snippets.
This function is defined in functions3 as:

similarity_search(query, directories) - Vectorstore embedding semantic search for code functions. It receives a query and the directories to search, and returns the most similar code snippets to the queries.

read_file(file_path) - Reads the contents of a file and returns it as a string.

write_to_file(content, file_path) - Writes given content to a specified file path, overwriting existing files.

pdb_tool(code_or_filename, action, line_number, variable_name) - Interacts with the Python debugger (PDB) to debug code.

Parameters:
    conversation (list): The conversation history
    functions3 (list): Helper functions for the API, including similarity_search
    max_response_tokens (int): Max tokens for the API response 
    token_limit (int): Max tokens allowed overall

Returns:
    None
    
The code improves the provided code snippets based on the conversation.

"""

# Define the initial messages in the conversation
functions3 = functions3
messages = [
    {
        "role": "system",
        "content": "You are a sophisticated AI that has the ability to analyze complex code and pseudocode documents. You are tasked with making necessary clarifications in a series of chat turns until you gather sufficient information to rewrite the code. You can utilize the 'search_code' function to fetch relevant code snippets based on semantic similarity, and subsequently improve the given file. After each search you should improve the file, do not make several calls to the function before improving the file."
    },
    {
        "role": "user",
        "content": f"search for openai functions calling 0613 using methaphor. after, scrape the second link"  
    }
]
conversation = messages




from tenacity import retry, wait_random_exponential, stop_after_attempt

# @retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(15))
# def api_call(messages, functions3, max_response_tokens):
#     try:
#         return openai.ChatCompletion.create(
#             model="gpt-4-0613",
#             messages=messages,
#             functions=functions3,
#             temperature=0.7,
#             max_tokens=max_response_tokens,
#             function_call="auto"
#         )
#     except openai.error.RateLimitError as e:
#         print(f"Rate limit exceeded: {e}")
#         raise
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#         raise


import time
import random
import openai

def api_call(messages, functions3, max_response_tokens):
    for i in range(15):
        try:
            return openai.ChatCompletion.create(
                model= "gpt-4",#"gpt-3.5-turbo-16k-0613",
                messages=messages,
                functions=functions3,
                temperature=0.7,
                max_tokens=max_response_tokens,
                function_call="auto"
            )
        except openai.error.RateLimitError as e:
            print(f"Rate limit exceeded: {e}")
            wait_time = 2 ** i + random.random()
            print(f"Waiting for {wait_time} seconds before retrying...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"An unexpected error occurred: {e if isinstance(e, str) else repr(e)}")

            raise
    print("Maximum number of retries exceeded. Aborting...")




def read_all_files_in_directory(directory_path):
    all_files_content = []
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                file_content = f.read()
            all_files_content.append(f"\n--- BEGIN {file_name} ---\n{file_content}\n--- END {file_name} ---")
    return ''.join(all_files_content)



def append_code_to_message(message, directory_path):
    words = message.split()
    file_names = [word[1:] for word in words if word.startswith("@")]
    appended_code = []

    for file_name in file_names:
        file_path = f"{directory_path}/{file_name}"
        file_content = read_file(file_path)

        if file_content is not None:
            appended_code.append(f"\n--- BEGIN {file_name} ---\n{file_content}\n--- END {file_name} ---")
        else:
            print(f"File {file_name} not found.")

    return f"{message}{''.join(appended_code)}"


while True:
    user_input = input("Enter message: ")

    if "@codebase" in user_input or "@" in user_input:
        directory_path = "/home/stahlubuntu/coder_agent/"

        # Replace @codebase tag with actual codebase content
        if "@codebase" in user_input:
            codebase_str = read_all_files_in_directory(directory_path)
            user_input = user_input.replace("@codebase", f"{codebase_str}")

        # Append the content of the files mentioned in the user_input
        user_input = append_code_to_message(user_input, directory_path)

    # Append the modified user_input to the conversation
    conversation.append({"role": "user", "content": user_input})

    # Check for slash commands
    if user_input == "/chat_history":
        pretty_print_conversation(conversation)
        continue
    elif user_input.startswith("/clean_chat_history"):
        if len(user_input) > 19:
            indices_str = user_input[20:].strip("[]").split(";")
            try:
                indices = [int(idx) for idx in indices_str]
                conversation = [msg for idx, msg in enumerate(conversation) if idx not in indices]
                print(f"Messages at indices {', '.join(indices_str)} have been removed!")
            except ValueError:
                print("Invalid format. Use /clean_chat_history [index1;index2;...]")
        else:
            conversation = []
            print("Chat history cleared!")
        continue
    elif user_input == "/token":
        tokens = num_tokens_from_messages(conversation)
        print(f"Current chat history has {tokens} tokens.")
        continue
    elif user_input == "/help":
        print("/chat_history - View the chat history")
        print("/clean_chat_history - Clear the chat history")
        print("/clean_chat_history [index1;index2;...] - Clear specific messages from the chat history")
        print("/token - Display the number of tokens in the chat history")
        print("/help - Display the available slash commands")
        continue
    
    conversation.append({"role": "user", "content": user_input})
    conv_history_tokens = num_tokens_from_messages(conversation)

    while conv_history_tokens + max_response_tokens >= token_limit:
        del conversation[1] 
        conv_history_tokens = num_tokens_from_messages(conversation)

    while True:
            
        chat_response = api_call(conversation, functions3, max_response_tokens)

        assistant_message = chat_response['choices'][0].get('message')
        finish_reason = chat_response['choices'][0].get('finish_reason')


        if assistant_message['content'] is not None:
            conversation.append({"role": "assistant", "content": assistant_message['content']})
            pretty_print_conversation(conversation)
        else:
            if assistant_message.get("function_call"):
                function_name = assistant_message["function_call"]["name"]
                arguments = json.loads(assistant_message["function_call"]["arguments"])

                # Implement the functions here
                if function_name == "similarity_search":
                    results = similarity_search(arguments['query'], arguments['directories'])
                    function_response = f"Code search content: {results}"
                elif function_name == "write_to_file":
                    # Assuming you have a write_to_file function somewhere
                    write_to_file(arguments['content'], arguments['file_path'])
                    function_response = f"File successfully written at {arguments['file_path']}"
                elif function_name == "read_file":
                    # Assuming you have a read_file function somewhere
                    content = read_file(arguments['file_path'])
                    function_response = content
                elif function_name == "ast_tool":
                    # Assuming you have an ast_tool function somewhere
                    function_response = ast_tool(arguments['code'], arguments['analyze_functions'], arguments['analyze_variables'], arguments['analyze_control_flow'])
                elif function_name == "pdb_tool":
                    # Assuming you have a pdb_tool function somewhere
                    result = pdb_tool(arguments['code_or_filename'], arguments['debug_operations'])
                    function_response = result
                elif function_name == "scrape_web_pages":
                    # Assuming you have a scrape_web_pages function somewhere
                    function_response = scrape_web_pages(arguments['urls'])

                elif function_name == "unit_test_runner":
                    # Assuming you have a unit_test_runner function somewhere
                    function_response = unit_test_runner(arguments['code_or_filename'], arguments['test_code_or_filename'])

                            
        

                elif function_name == "automated_code_reviewer":
                    # Assuming you have an automated_code_reviewer function somewhere
                    feedback = automated_code_reviewer(arguments['code_or_filename'])
                    function_response = feedback


                elif function_name == "terminal_access":
                    # Assuming you have an automated_code_reviewer function somewhere
                    feedback = terminal_access(arguments['command'])
                    function_response = feedback

                # elif function_name == "dynamic_code_execution":
                #     # Assuming you have an automated_code_reviewer function somewhere
                #     feedback = dynamic_code_execution(arguments['code_or_filename'], arguments['mode'], arguments['input_var'], arguments['return_vars'])
                #     function_response = feedback

                elif function_name == "dynamic_code_execution":
                    if 'input_var'  in arguments:
                        input_var = arguments['input_var'] 
                    else:
                        input_var = None
                    if 'return_vars' in arguments:
                        return_vars = arguments['return_vars']
                    else:
                        return_vars = None
                    if 'mode' in arguments:
                        mode = arguments['mode']
                    else:
                        mode = None
                    feedback = dynamic_code_execution(arguments['code_or_filename'], mode, input_var, return_vars)
                    function_response = feedback

                elif function_name == "metaphor_web_search":
                    # Assuming you have an automated_code_reviewer function somewhere
                    #feedback = metaphor_web_search(arguments['query'], arguments['num_results'], arguments['start_published_date'], arguments['end_published_date'])
                    if 'num_results' in arguments:
                        num_results = arguments['num_results']
                    else:
                        num_results = 10
                    if 'start_published_date' in arguments:
                        start_date = arguments['start_published_date']
                    else:
                        start_date = None

                    if 'end_published_date' in arguments:
                        end_date = arguments['end_published_date']
                    else:
                        end_date = None

                    feedback = metaphor_web_search(arguments['query'], start_date, end_date)
                    function_response = feedback

                elif function_name == "complexity_analyzer":
                    if 'code_or_filename' in arguments:
                        code_or_filename = arguments['code_or_filename']
                    else:
                        raise ValueError("Missing required argument 'code_or_filename'")

                    if 'include_comments' in arguments:
                        include_comments = arguments['include_comments']
                    else:
                        include_comments = False

                    if 'include_whitespace' in arguments:
                        include_whitespace = arguments['include_whitespace']
                    else:
                        include_whitespace = False

                    complexity = complexity_analyzer(code_or_filename, include_comments, include_whitespace)
                    function_response = complexity

                elif function_name == "code_profiler":
                    if 'code_or_filename' in arguments:
                        code_or_filename = arguments['code_or_filename']
                    else:
                        raise ValueError("Missing required argument 'code_or_filename'")
                    
                    if 'sort_by' in arguments:
                        sort_by = arguments['sort_by']
                    else:
                        sort_by = 'cumulative'
                    
                    if 'limit' in arguments:
                        limit = arguments['limit']
                    else:
                        limit = 10
                    
                    profiling_data = code_profiler(code_or_filename, sort_by, limit)
                    function_response = profiling_data


                # Add assistant and function response to conversation
                conversation.append({
                    "role": "assistant",
                    "function_call": {
                        "name": function_name,
                        "arguments": assistant_message["function_call"]["arguments"]
                    },
                    "content": None
                })
                conversation.append({
                    "role": "function",
                    "name": function_name,
                    "content": function_response
                })

                # Optional: Make a second API call to get the final assistant response
                second_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0613",
                    messages=conversation,
                    functions=functions3
                )

                # Second API call to get the final assistant response
                #chat_response = api_call(conversation, functions3, max_response_tokens)
                final_message = second_response["choices"][0]["message"]
                if final_message['content'] is not None:
                    conversation.append({"role": "assistant", "content": final_message['content']})
                    pretty_print_conversation(conversation)

        print(f"Finish reason: {finish_reason}")
        if finish_reason != 'function_call':
            print("Breaking the inner loop.")
            break

            

