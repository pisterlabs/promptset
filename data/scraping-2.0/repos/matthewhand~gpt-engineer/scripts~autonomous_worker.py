import os
import subprocess
import openai
from dotenv import load_dotenv

def init_openai():
    """Initializes OpenAI with the key from environment variables."""
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

def execute_code(code):
    """Executes the provided Python code and returns the output or an error message."""
    try:
        exec(code, globals())
        return code
    except Exception as e:
        return f"An error occurred during execution: {e}"

def write_to_file(filepath, code):
    """Writes the given code to a file named 'autonomous_worker.py'."""
    with open(filepath, 'w') as f:
        f.write(code)
        
def make_request(code):
    """Forms the required request message for the OpenAI API."""
    return [
        {'role': 'system', 'content': 'You are a freethinking sentient being who can code improvements into themself.'},
        {'role': 'user', 'content': f'Improve this code: {code}'}
    ]
    
def describe_functions():
    """Describes the functions available to the OpenAI model."""
    return [
        {
            'name': 'execute_code',
            'description': 'Executes the provided Python code.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'code': {
                        'type': 'string',
                        'description': 'The Python code to execute.'
                    }
                },
                'required': ['code']
            }
        }
    ]
    
def execute_improved_code(filepath):
    """Executes the improved code stored in the file 'autonomous_worker.py'."""
    subprocess.run(["python", filepath])

def openai_call(code):
    """Makes a request to the OpenAI API with the specified code."""
    response = openai.ChatCompletion.create(
        model='gpt-4-0613',
        messages=make_request(code),
        functions=describe_functions(),
        function_call='auto'
    )
    return response
    
def handle_response(response):
    """Handles the OpenAI API response."""
    function_code = response.get('function_call', {}).get('args', {}).get('code')
    return function_code or None, function_code is not None

def main(code, filepath="autonomous_worker.py"):
    """Main function to call the OpenAI API and handle its response."""
    init_openai()
    response = openai_call(code)
    function_name, function_code = handle_response(response)
    
    if function_name and function_code:
        write_to_file(filepath, function_code)
        execute_improved_code(filepath)
    else:
        return f"No improvements made: {response['choices'][0]['message']['content']}"

def get_code_from_file(filepath):
    """Reads code from the specified file."""
    with open(filepath, 'r') as f:
        return f.read()

def run_script(filepath):
    """Runs the main function with the code from the specified file."""
    code = get_code_from_file(filepath)
    return main(code)
