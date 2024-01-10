import os
import argparse
import numpy as np
import requests
import openai
import subprocess
import json
import time
import re
from termcolor import colored


class Code:
    def __init__(self, code='', output=None, error=None, path=None):
        self.code = code
        self.code_lines = code.split('\n')
        self.output = output
        self.error = error
        self.fixes = []
        self.path = ''

        if path is not None:
            self.from_file(path)

    def to_json(self):
        return {
            'code': self.code,
            'output': self.output,
            'error': self.error
        }

    @staticmethod
    def from_json(json_obj):
        return Code(json_obj['code'], json_obj['output'], json_obj['error'])

    def fix(self, lines, new_code):
        new_code = new_code.split('\n')
        if len(lines) == 1:
            lines = [lines[0], lines[0]]
        elif len(lines) == 0:
            if new_code == '':
                print('No fix is needed')
                return
            else:
                print('A new code was given, but instructions of where to put it are missing.')
                return
        j = 0
        for i in range(lines[0] - 1, lines[1]):
            self.code_lines[i] = new_code[j]
            j += 1
        self.compile_code()

    def compile_code(self):
        self.code = '\n'.join(self.code_lines)

    def debug(self, model):
        input_json = self.to_json()
        prompt = f"""
        You are a code debugger. Here is the code, its output, and the error it produced:

        {json.dumps(input_json, indent=4)}

        Please identify the lines that need to be changed and suggest the new code to fix the issue. 
        Return your response in the following JSON format:

        {{
            "lines": [start_line, end_line],
            "new_code": "the new code"
        }}

        Note to yourself:
        - If there is only one line to be changed, the value on the key "lines", will be as [change_line, change_line], i.e both elements of the list will be the same single line.
        - Add nothing else to you response, send only the JSON.
        - The content of this prompt might be divided into a few parts, and be sent in a sequence. 
        Therefore, you should not send any response back, until you receive the total prompt. To know when the prompt is complete,
        expect the total content of this complete prompt to end with only the JSON with keys {{'code','output','error'}}.
        """

        prompt_parts = [prompt[i:i + 4097] for i in range(0, len(prompt), 4097)]
        responses = []
        for part in prompt_parts:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": part}
                ]
            )
            responses.append(response['choices'][0]['message']['content'])

        content = ''.join(responses)

        try:
            # Use a regex to extract the JSON object from the response
            match = re.search(r'\{\s*"lines":\s*\[.*\],\s*"new_code":\s*".*"\s*\}', content, re.DOTALL)
            if match:
                json_str = match.group(0)
                response_json = json.loads(json_str)
            else:
                print("No JSON object found in the response.")
                response_json = {'lines': [], 'new_code': ''}
        except json.JSONDecodeError:
            print("The content could not be parsed as JSON.")
            response_json = {'lines': [], 'new_code': ''}

        self.fix(response_json["lines"], response_json["new_code"])

    def to_file(self):
        path = self.path
        # Rename the old file if it exists
        if os.path.exists(path):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            os.rename(path, f"{path}_{timestamp}")

        with open(path, "w") as f:
            for line in self.code:
                f.write(line)

    def from_file(self, path):
        self.path = path
        # Read the code from the file
        with open(self.path, 'r') as f:
            self.code = f.read()

    def run(self, env_name, args=''):
        args = args.split(' ')
        try:
            # Run the Python file in the specified conda environment
            result = subprocess.run(['conda', 'run', '-n', env_name, 'python', self.path] + args, capture_output=True,
                                    text=True)
            self.output = result.stdout
            self.error = result.stderr
        except Exception as e:
            self.output = ''
            self.error = str(e)



def run_python_file(file_path, env_name, args=''):
    args = args.split(' ')
    try:
        # Run the Python file in the specified conda environment
        result = subprocess.run(['conda', 'run', '-n', env_name, 'python', file_path] + args, capture_output=True,
                                text=True)
        output = result.stdout
        error = result.stderr
    except Exception as e:
        output = ''
        error = str(e)

    # Read the code from the file
    with open(file_path, 'r') as f:
        code = f.read()

    return Code(code, output, error, path=file_path)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help="Path to file")
    parser.add_argument('--env', help="Name of conda environment", default="utils")
    parser.add_argument('--model', help="Name of GPT model", default='gpt-3.5-turbo')
    parser.add_argument('--n', help="Max number of iterations", default=3)
    parser.add_argument('--args', help="Default args", default=[])
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Missing OpenAI API key")
    openai.api_key = api_key
    return parser.parse_args()


def main():
    args = parse()
    j = 0
    args.file = 'SR2.py'
    args.args = '--train'

    for i in range(args.n):
        j = i

        code = run_python_file(args.file, args.env, args.args)
        print(colored(f'code:\n{code.code}', 'yellow'))
        print(colored(f'output:\n{code.output}', 'blue'))
        print(colored(f'error:\n{code.error}', 'red'))

        if code.error == '':
            break

        code.debug(args.model)
        code.to_file()

    print(f"All went well. It took {j + 1} runs.")


if __name__ == '__main__':
    main()
