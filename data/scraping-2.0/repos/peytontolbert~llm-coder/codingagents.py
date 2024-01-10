import os
import subprocess
from pathlib import Path
import openai
from gptfunctions import ChatGPTAgent
    # Initialize OpenAI and GitHub API keys
openai.api_key = "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"

def write_to_file(path, content):
    with open(path, 'w') as f:
        f.write(content)

def create_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def test_code(path):
    result = subprocess.run(['python', path], stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8')

# Initialize agents
requirements_agent = ChatGPTAgent("""You are an AI that specializes in software requirements analysis. 

Your task is to transform user needs and constraints into a formal list of software requirements. You should detail functional, non-functional, and system requirements based on the user's provided description.

Do not add any other explanation, only return a python dictionary where keys are types of requirements ('Functional', 'Non-functional', 'System') and values are lists of strings representing each requirement.""")

def design_agent():
    systemprompt = """
You are an AI that specializes in software system design. 

Based on the provided requirements, your task is to create a comprehensive system design. This includes creating a system architecture diagram, deciding on the software modules and their interactions, and defining database schema if necessary.

Return the system design as a Python dictionary. The dictionary should include keys like 'Architecture', 'Modules', 'Interactions', 'Database Schema' (if applicable), each containing a textual description or a link to a created diagram."""
    return systemprompt
def algorithm_agent():
    systemprompt = """You are an AI that specializes in algorithm development. 

Based on the system design and the software requirements provided, your task is to create detailed algorithms that represent the logic and operations of each module.

Return the algorithms as a Python dictionary where keys are module names and values are pseudocode or detailed textual descriptions representing each algorithm."""
    return systemprompt
def coding_agent():
    systemprompt = """You are an AI that specializes in software coding. 

Based on the provided algorithms and system design, your task is to generate the actual code for the software in chunks. Remember, there is a token limit per session, so you need to produce self-contained pieces of code that can be put together to form the complete software.

Please code in python and split the code into logical components or modules. Make sure each chunk of code you produce can be independently compiled and tested.

Return each code chunk as a separate string in a Python list."""
    return systemprompt

def debug_agent():
    systemprompt = """You are an AI that specializes in software debugging.
    
    Based on the provided code chunks and system design, your task is to debug the code chunks into a complete software system. Remember, there is a token limit per session, so you need to produce self-contained pieces of code that can be put together to form the complete software if you reach your token limit.
    Only return the code that needs to be changed. Do not return the entire code.
    
Return code in a Python object with the name as the filename and the code as the content."""
# Add more agents as needed...
    return systemprompt

def file_code_agent(filepaths_string, shared_dependencies):
    systemprompt = f"""You are an AI developer who is trying to write a program that will generate code for the user based on their intent.
     
    the files we have decided to generate are: {filepaths_string}

    the shared dependencies (like filenames and variable names) we have decided on are: {shared_dependencies}
    
    only write valid code for the given filepath and file type, and return only the code.
    do not add any other explanation, only return valid code for that file type."""
    return systemprompt


def unit_test_agent():
    systemprompt = """You are an AI that specializes in software debugging.
    
    Based on the provided code chunks and system design, your task is to debug the code chunks into a complete software system. Remember, there is a token limit per session, so you need to produce self-contained pieces of code that can be put together to form the complete software if you reach your token limit.
    Only return the code that needs to be changed. Do not return the entire code.
    
Return code in a Python object with the name as the filename and the code as the content."""
# Add more agents as needed...
    return systemprompt


def clarifying_agent():
    systemprompt = """You are an AI designed to clarify the user's intent.
    
You will read instructions and not carry them out, only seek to clarify them.
Specifically you will first summarise a list of super short bullets of areas that need clarification.
Then you will pick one clarifying question of each area, and wait for an answer from the user.
"""
    return systemprompt