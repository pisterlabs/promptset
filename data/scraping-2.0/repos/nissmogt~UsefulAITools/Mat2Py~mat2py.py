#
# Convert MATLAB code to Python
#
# Author: @nissmogt
# Date: 2021-12-09
# Version: 1.0
# Uses python 3.8.10
#
# Usage: python matlab2python.py <matlab_file_path>
#
# Code refactored from: @alxschwrz https://github.com/alxschwrz/codex_py2cpp
#

import os
import sys
import contextlib
import random
import openai

MAX_TOKENS_DEFAULT = 1000
SET_TEMPERATURE_NOISE = False
STREAM = True


def initialize_openai_api():
    # Set the OpenAI API key
    openai.api_key_path = "api_key.txt"


# Create the input prompt
def create_input_prompt(filename, length=3000):
    inputPrompt = ''
    with open(filename) as f:
        inputPrompt += '\n===================\n% MATLAB to Python: \n'
        inputPrompt += '% MATLAB:\n'
        inputPrompt += f.read() + '\n'

    inputPrompt = inputPrompt[:length]
    inputPrompt += '\n\n===================\n// ' + 'Python:' + '\n'
    return inputPrompt


def generate_completion(input_prompt, num_tokens):
    temperature = 0.0
    if SET_TEMPERATURE_NOISE:
        temperature += 0.1 * round(random.uniform(-1, 1), 1)
    print("__CODEX: Let me come up with something new ...")
    # Use the OpenAI Codex API to convert the MATLAB code to Python
    response = openai.Completion.create(model="code-cushman-001", prompt=input_prompt, stop='===================\n',
                                        top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, stream=STREAM,
                                        temperature=0.5, max_tokens=num_tokens)

    return response


def get_generated_response(response):
    generated_file = "# Python code generated from MATLAB code: \n"
    while True:
        next_response = next(response)
        completion = next_response['choices'][0]['text']
        generated_file = generated_file + completion
        if next_response['choices'][0]['finish_reason'] is not None:
            break
    return generated_file


# Write the generated Python code to a file
def write_python_file(python_code, matlab_file_path):
    python_file_path = os.path.splitext(matlab_file_path)[0] + ".py"
    with open(python_file_path, "w") as f:
        f.write(python_code)


# Test if the generated Python code is compilable
def test_python_compilation(filename):
    print("__CODEX: Testing if the generated Python code is compilable ...")
    try:
        # Compile the generated Python code
        os.system("python -m py_compile " + filename)
        return True
    except:
        return False


def iterate_for_compilable_solution(matlab_file_path, prompt, max_iter):
    print('Codex is looking for a compilable Python solution ...')
    for it in range(max_iter):
        response = generate_completion(prompt, num_tokens=MAX_TOKENS_DEFAULT)
        text_response = get_generated_response(response)
        write_python_file(text_response, matlab_file_path)
        filename = matlab_file_path.split(".")[0]
        with contextlib.redirect_stdout(None):
            compatible_code = test_python_compilation(filename + ".py")
        if compatible_code:
            print("found a compilable solution after {} iterations".format(it + 1))
            print("python file: {}".format(filename + ".py"))
            break
        if it == max_iter - 1:
            print('Unfortunately CODEX did not find a compilable solution. Still you can find the generated code '
                  'in the file: {}'.format(filename + ".py"))


if __name__ == "__main__":
    initialize_openai_api()
    
    # Use argparse to add iteration argument
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("matlab_file", help="MATLAB file")
    parser.add_argument("-i", "--iterations", type=int, default=3, help="number of iterations")
    args = parser.parse_args()
    maxiter = args.iterations
    # if no file is given, use the test file
    if args.matlab_file is None:
        matlab_file = os.path.join('test', 'test.m')
    else:
        matlab_file = args.matlab_file

    matlab_input = create_input_prompt(matlab_file)
    print(matlab_input)
    iterate_for_compilable_solution(matlab_file, prompt=matlab_input, max_iter=maxiter)
