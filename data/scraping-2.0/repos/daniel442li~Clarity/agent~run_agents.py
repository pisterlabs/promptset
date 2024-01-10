import subprocess
import re
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv, find_dotenv
from pyfiglet import figlet_format
from termcolor import colored, cprint
from logs import log
import questionary
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
from openai import OpenAI
import concurrent.futures
import json 
from collections import OrderedDict

load_dotenv(find_dotenv())
client = OpenAI()

from prompt_toolkit.styles import Style

custom_style_fancy = Style([
    ('qmark', 'fg:#e770ff bold'),       # token in front of the question
    ('question', 'bold'),               # question text
    ('answer', 'fg:#e770ff bold'),      # submitted answer text behind the question
    ('pointer', 'fg:#e770ff bold'),     # pointer used in select and checkbox prompts
    ('highlighted', 'fg:#e770ff bold'), # pointed-at choice in select and checkbox prompts
    ('selected', 'fg:#e770ff'),         # style for a selected item of a checkbox
    ('separator', 'fg:#e770ff'),        # separator in lists
    ('instruction', ''),                # user instructions for select, rawselect, checkbox
    ('text', ''),                       # plain text
    ('disabled', 'fg:#e770ff italic')   # disabled choices for select and checkbox prompts
])  

def clear(): 
    if os.name == 'nt': 
        _ = os.system('cls') 
    else: 
        _ = os.system('clear') 


def run_tests():
    # Define the path to your shell script
    shell_script = './run_tests.sh'

    # Run the shell script with subprocess.Popen and capture the output
    proc = subprocess.Popen(shell_script, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    # Wait for the process to terminate and get the stdout and stderr
    stdout, stderr = proc.communicate()

    # Decode the stdout to string from bytes, if necessary
    output = stdout.decode('utf-8')
    # Use a regular expression to search for the number of passing tests in the output
    match = re.search(r'Number of tests passed: (\d+)', output)

    if match:
        # Extract the number of passing tests
        passing_tests = match.group(1)
        
    else:
        # If there is no match, output an error message
        print("Could not extract the number of passing tests.")
        print(f"Error output: {stderr.decode('utf-8')}")

    passing_tests_num = int(passing_tests) if passing_tests else None
    return output, passing_tests_num


def call_gpt(prompt):
    response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
        {"role": "system", "content": "You are a software engineer writing a file. Please make the function fully functional for CommonJS. Do not import any additional libraries."},
        {"role": "user", "content": prompt}]
    )
    
    result = (response.choices[0].message.content)
    return result

pattern = re.compile(r'```javascript\s*([\s\S]+?)\s*```')
pattern2 = re.compile(r'```js\s*([\s\S]+?)\s*```')
# Reading the entire content of the .cjs file at once
with open('./src/server/api/v1/example_user.cjs', 'r', encoding='utf-8') as file:
    editable_file = file.read()


def generate_oneshot_code(files):
    cprint(figlet_format('Running AI on Codebase', font='digital'), 'magenta')
    supporting_files = ""

    log("Using these files: " + str(files), "info")
    for file_path in files: 
        if file_path == 'No Context':
            continue
        with open(file_path, 'r', encoding='utf-8') as file:
            supporting_files += "\n" + file_path + "\n" + file.read()
            
    template_string = f'''

    The file you will be working in is called `user.cjs`.
    {editable_file}

    Relevant files inside of the code base (tests, additional functions)
    {supporting_files}
    '''
    
    log("Sending context to AI", "info")
    results = []

    # Using ThreadPoolExecutor to call the function in multiple threads.
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Create a list to hold the futures.
        futures = [executor.submit(call_gpt, template_string) for _ in range(3)]

        # Wait for all futures to complete and add the result to the results list.
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    for result in results:
        match = pattern.search(result)
        if match:
            log("Code created by AI", "info")
            edited_code = match.group(1)
        else:
            match = pattern2.search(result)
            if match:
                log("Code created by AI", "info")
                edited_code = match.group(1)
            else: 
                log("Code errored out. Please check logs", "info")
                print(result)

        # Reading the entire content of the .cjs file at once# Specify the path to the file
        file_path = './src/server/api/v1/user.cjs'

        # Check if the file exists and delete it
        if os.path.exists(file_path):
            os.remove(file_path)

        # Now create a new file at the same location
        with open(file_path, 'w') as file:
            file.write(edited_code)
            log(f"Created a new file: {file_path}", "info")

        
        output, passing_tests = run_tests()

        with open('./agent/results.json', 'r') as json_file:
            result_dict = json.load(json_file)

        result_dict[str(files)] = result_dict.get(str(files), []) + [passing_tests]

        sorted_items = sorted(result_dict.items(), key=lambda item: len(item[0]))

        sorted_dict = OrderedDict(sorted_items)

        with open('./agent/results.json', 'w') as json_file:
            json.dump(sorted_dict, json_file, indent=4)

        log(f"Number of tests passed: {passing_tests}", "results")

    while True:
        choice = questionary.select(
        "Additional Actions",
            choices=["Print Testing Output", "Print Generated Code and Reasoning", "Print Prompt", "Return to Main Menu"],
        ).ask()
        
        if choice == "Print Testing Output":
            print(highlight(output, PythonLexer(), TerminalFormatter()))
        if choice == "Print Generated Code and Reasoning":
            print(highlight(result, PythonLexer(), TerminalFormatter()))
        
        if choice == "Print Prompt":
            print(highlight(template_string, PythonLexer(), TerminalFormatter()))

        if choice == "Return to Main Menu":
            clear()
            break


        
    return passing_tests


