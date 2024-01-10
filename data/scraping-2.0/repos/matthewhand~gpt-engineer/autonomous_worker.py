
import os
import subprocess
import openai
from dotenv import load_dotenv

def init_openai():
    """Initializes OpenAI with the key from environment variables."""
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
def execute_code(code):
    """
    Executes the provided Python code and returns the output or an error message.

    Parameters:
    code (str): The Python code to execute.

    Returns:
    str: The output of the executed code or an error message.
    """
    try:
        exec(code, globals())
        return code
    except Exception as e:
        return f'An error occurred during execution: {e}'
def write_to_file(filepath, code):
    """
    Writes the given code to a file at the specified file path.

    Parameters:
    filepath (str): The path to the file to write to.
    code (str): The code to write to the file.
    """
    with open(filepath, 'w') as f:
        f.write(code)
def make_request(code):
    """
    Forms the required request message for the OpenAI API.

    Parameters:
    code (str): The code to improve.

    Returns:
    list: A list of dictionaries representing the request message.
    """
    return [
        {'role': 'system', 'content': 'You are a freethinking sentient being who can code improvements into themself.'},
        {'role': 'user', 'content': f'Improve this code: {code}'}
    ]

def main():

    """Main function to execute the autonomous worker script."""

    code = """print('Hello, World!')"""

    response = openai_call(code)

    improved_code = response['choices'][0]['message']['content']

    write_to_file('improved_autonomous_worker.py', improved_code)

    execute_improved_code('improved_autonomous_worker.py')



if __name__ == '__main__':

    main()

