import openai
import os
import sys
import subprocess
import shlex
import requests
from termcolor import colored, cprint
import json
import random

openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = """
Instructions: You are a helpful assistant who, for a given natural language request, generates a Bash script that executes the task requested. Assume the computer is running Mac OS. Do not respond with anything other than the Bash script. Do not talk to the user. Any output that is not part of the Bash script must be commented out. If the request is missing critical information, create Bash variable(s) called $VAR_X (where X is replaced by incrementing integers for each new variable), and for each variable explain which information is needed to fill it, preceded by the string "NEEDS_INFO" Do not perform any "read" statements in Bash, just insert the variables wherever needed in the script.

A few examples of valid requests and responses:

Request: How do I write a limerick and save it to the Documents folder?
Output:
echo "There was an Old Man with a beard, Who said, 'It is just as I feared! Two Owls and a Hen, Four Larks and a Wren, Have all built their nests in my beard!" > ~/Documents/limerick.txt

Request: generate a git repository.
Output:
cd $VAR_1
mkdir $VAR_2
cd $VAR_2
git init
NEEDS_INFO VAR_1 where should the Git repository be generated?
NEEDS_INFO VAR_2 what should the new directory be named?

Request: print the fibonacci numbers.
for (( i=0; i<$VAR_1; i++ ))
do
    echo -n "$a "
    fn=$((a + b))
    a=$b
    b=$fn
done
NEEDS_INFO VAR_1 how many Fibonacci numbers should be printed?

Request: {{command}}. Please write a Bash script step-by-step.
Output:
"""

explain_prompt = """
You are a helpful assistant who, for a given Bash script, summarizes the actions that are being taken by that Bash script. Be as concise as possible. A few examples include:

Script:
mkdir my-project
cd my-project
npm init -y
npm install express --save
Output:
Creating a folder called "my-project" in the current directory, initializing an NPM package, and installing express.

Script:
open -a "Safari" "https://www.google.com/search?q=5th+tallest+mountain+in+the+world"
Output:
Opening Safari and Googling "5th tallest mountain in the world".

Script:
{{script}}
Output:
"""

def gen_uid():
    return ''.join(random.choice('0123456789abcdef') for i in range(16))

def get_bash_explanation(query):
    messages = [
        {
            'role': 'user',
            'content': explain_prompt.replace('{{script}}', query)
        }
    ]

    result = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        stop=None,
        temperature=0,
    )['choices'][0]['message']['content']

    return result

def perform_bash_command(query):
    messages = [
        {
            'role': 'user',
            'content': prompt.replace('{{command}}', query)
        }
    ]

    script = None
    script_fname = '/tmp/script-' + gen_uid() + '.sh'

    def generate_script():
        needinfo_questions = {}
        require_vars = {}

        result = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=messages,
            stop=None,
            temperature=0,
        )['choices'][0]['message']['content']

        script_lines = []

        messages.append({
            'role': 'assistant',
            'content': result
        })

        # Format the result of the completion
        lines = result.strip().split('\n')

        for line in lines:
            if line.startswith('NEEDS_INFO'):
                var = line.split(' ')[1]
                needinfo_questions[var] = line.split(' ', 2)[2]
            else:
                script_lines.append(line)

        script = '\n'.join(script_lines)

        cprint('\n' + script + '\n', 'dark_grey')

        # Ask for the variables
        for var, question in needinfo_questions.items():
            cprint(f'PROMPT ({var}): ' + question + ' > ', attrs=['bold'], end='')
            value = input()
            require_vars[var] = value

        # Replace variables
        for var, value in require_vars.items():
            script = script.replace('$' + var, value)

        # Save the script to a temporary file
        with open(script_fname, 'w') as f:
            f.write('set -e\nset -o pipefail\n' + script + '\nset +o pipefail\nset +e')

        return script

    script = generate_script()

    # Execute the script
    has_error = True
    retries = 0
    error_output = None

    while has_error and retries < 3:
        try:
            output = subprocess.check_output(['bash', script_fname], stderr=subprocess.STDOUT)
            has_error = False
        except subprocess.CalledProcessError as e:
            err_string = e.output.decode('utf-8')
            error_output = err_string
            messages.append({
                'role': 'user',
                'content': f'Oh no! The script failed with the following error:\n\n{err_string}\n\n' + \
                    f'Rewrite the script from scratch fix the error.\nRequest: How do I {query}?\n' + \
                    'Output:'
            })
            cprint('The script failed. Rewriting...\n', 'red')
            script = generate_script()
            retries += 1

    if has_error:
        cprint('The script failed too many times. Please try again later.\n', 'red')
        cprint(error_output, 'red')
        sys.exit(1)

    # Print the output
    cprint('\n' + get_bash_explanation(script) + '\n', attrs=['bold'])
    cprint('Output:' + '\n', 'dark_grey')
    cprint(output.decode('utf-8') + '\n', attrs=['bold'])

    # Delete the temporary file
    os.remove(script_fname)


def please(user_input):
    perform_bash_command(user_input)

if __name__ == "__main__":
    please()
