#!/usr/bin/env python3
"""
    autodoc.py

    Usage:
        autodoc.py <my_code.py>

    NOTE: This requires Python 3.9+ (this is openai's library requirement).

    This app is a work in progress.
    The long-term plan is to add docstrings to the functions and classes in the
    file `my_code.py` provided on the command line.

    Currently the modified file is printed to stdout.
    This finds all function and method definitions in the input file and adds a
    docstring for them. This currently assumes there are no docstrings for such
    definitions. If there are already docstrings, then you will end up with the
    new, gpt-based docstring on top of the old docstring.

    The wishlist / todo ideas below clarify more about what the current v1
    script does _not_ do. :)
"""

# TODO Ideas:
#
#  * Work nicely with existing docstrings (use as input and overwrite).
#  * Add docstrings for class definitions.
#  * Detect indentation size type per file.
#  * Wrap long lines at detected file width (or command-line param).
#  * We could add per-line or per-code-paragraph comments.
#  * Aim for consistent style in terms of where newlines are and quotes are for
#    each docstring. Eg, is the first line of docstring content on the same line
#    as the opening quotes? Is the content indented? Etc.
#  * Consider using a flag to replace/augment print_to_console setting in config
#    file.

# KNOWN BUGS:
#  * Actually measure the number of tokens before we send requests in order to
#    avoid sending requests that involve too many tokens for GPT.


# ______________________________________________________________________
# Imports (OpenAI is imported only after all-systems-are-a-go farther below)

# Standard library imports.
import json
import os
import random
import re
import shutil
import sys
import time
from inspect import cleandoc
from pathlib import Path


# ______________________________________________________________________
# Constants and globals

# This is the maximum byte length of code that we'll send to GPT in one request.
# Really, we ought to be measuring the number of _tokens_. This is a rough
# approximation that can fail, but will often work in practice.
# TODO: Use this everywhere, not just for top-of-file docstrings.
MAX_CODE_STR = 9000

NUM_REPLY_TOKENS = 700

output_file = None

# Turn this on to have additional debug output written to a file.
if True:
    dbg_f = open('dbg_out.txt', 'w')
else:
    dbg_f = None


# ______________________________________________________________________
# Debug functions

def pr(s=''):
    global dbg_f
    if dbg_f:
        print(s, file=dbg_f)


# ______________________________________________________________________
# GPT functions

def send_prompt_to_gpt(prompt):
    """
    This function sends the provided prompt to GPT and returns GPT's response.
    """

    # Document what's happening to the debugger output file
    pr('\n' + ('_' * 70))
    pr('send_prompt()')
    pr(f'I will send over this prompt:\n\n')
    pr(prompt)

    if MOCK_CALLS:
        gpt_response = ('\nTHIS IS A MOCK DOCSTRING. To change this, ' +
                        'set "mock_calls" to false in config.json.\n"""')
    else:
        # Send request to GPT, return response
        response = openai.Completion.create(
            model             = 'text-davinci-003',
            prompt            = prompt,
            temperature       = 0,
            max_tokens        = NUM_REPLY_TOKENS,
            top_p             = 1.0,
            frequency_penalty = 0.0,
            presence_penalty  = 0.0
        )
        gpt_response =  response['choices'][0]['text']

    return '"""' + gpt_response


def fetch_docstring(code_str):

    # Construct the GPT prompt
    prompt  = 'Write a docstring for the following code:\n\n'
    prompt += code_str[:MAX_CODE_STR]
    prompt += '\n\nDocstring:\n"""'

    # Make the request for docstring to GPT
    docstring = send_prompt_to_gpt(prompt)

    # Document what's happening to the debugger output file
    pr('Got the docstring:\n')
    pr(docstring)

    # Return it
    return docstring

# ______________________________________________________________________
# Print Functions

def print_out(line):
    """
        This function prints the updated code in one of two ways:
            - to a file in the output directory
            - to console
    """
    if not PRINT_TO_CONSOLE:
        output_file.write(line)
        output_file.write('\n')
    else:
        print(line)


def print_fn_w_docstring(code_str):
    """ 
        This function requests GPT provide a docstring for the function code
        (as a str) provided as an argument.  It then prints the function with
        the docstring added.
    """

    # Fetch the docstring.
    docstring = fetch_docstring(code_str)

    # Print the function header/signature.
    code_lines = code_str.split('\n')
    print_out(code_lines[0])

    # Print the docstring.
    indent = re.search(r'^(\s*)', code_lines[0])
    indent = len(indent.group(1))
    prefix = ' ' * (indent + 4)

    for ans_line in docstring.split('\n'):
        print_out(prefix + ans_line)

    # Print the rest of the function.
    for line in code_lines[1:]:
        print_out(line)

# This only prints messages to standard out if we aren't printing the modified
# code output to the console.
def print_status_msg(msg, end='\n', flush=False):
    if not PRINT_TO_CONSOLE:
        print(msg, end=end, flush=flush)

# ______________________________________________________________________
# Main

if __name__ == '__main__':

    # If the config file does not exist, create it from the template.
    keyfile = Path('config.json')
    if not keyfile.is_file():
        shutil.copyfile('templates/config.template', 'config.json')

    # Open the config file
    with keyfile.open() as f:
        config = json.load(f)

    # Verify config.json contains a non-null definition for the API key.
    if not ('api_key' in config and config['api_key']):
        print(cleandoc('''
            Error: You're missing a openai API key in config.json.
            Please set {"api_key": "YOUR_API_KEY"} where YOUR_API_KEY is the
            key you generate at https://beta.openai.com/account/api-keys.
        '''))
        sys.exit(0)

    # Use the config data.
    OPENAI_API_KEY = config['api_key']
    PRINT_TO_CONSOLE = config['print_to_console'] if ('print_to_console' in config) else False
    MOCK_CALLS = config['mock_calls'] if ('mock_calls' in config) else False


    # If this script has been improperly executed, print the docstring & exit.
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    # Open and ingest the Python file provided as an input.
    input_filename_path = sys.argv[1]
    input_filename = input_filename_path.split('/')[-1]
    output_file_path = f'output/{input_filename}'

    with open(input_filename_path) as f:
        code = f.read()
    lines = code.split('\n')

    # If appropriate, inform the user that mock_calls is turned on
    if MOCK_CALLS:
        print_status_msg(cleandoc('''
            Note: Calls to GPT will be mocked. (To change this, open config.json
            and change "mock_calls" to false.)
        ''') + '\n')
    else:
        # The openai library is slow to load, so be clear something is happening.
        print_status_msg('Loading OpenAI library .. ', end='', flush=True)
        # Third party imports.
        import openai
        print_status_msg('done!')
        openai.api_key = OPENAI_API_KEY

    # If our output is going to a file, create and open a file in the output
    # directory by the same name, for writing. 
    if not PRINT_TO_CONSOLE:
        # Ensure the output directory exists.
        Path('output').mkdir(exist_ok=True)
        output_file = open(output_file_path, 'w')

    #######################################
    # BEGIN GENERATING CODE WITH DOCSTRINGS
    #######################################

    # Get the 'Top of File' docstring.
    print_status_msg('Writing top-of-file docstring .. ', end='', flush=True)
    tof_docstring = fetch_docstring(code)
    print_status_msg('done!')
        
    # Print Out Input Code with Docstrings Inserted
    #       Walk through the input code, line-by-line.
    #       Add the top-of-file docstring.
    #       Then print out code, until you find a function. 
    #       When you find a function, capture it and have GPT provide a
    #       docstring for it.
    #       Print out the function with docstring.
    #       Continue as before until file end.

    # Print out any shebang line as a special case.
    if lines[0].startswith('#!'):
        print_out(lines[0])
        lines = lines[1:]

    # Print out the Top-of-File Docstring
    print_out(tof_docstring)

    # Set up vars for capturing functions    
    capture_mode = False
    indentation  = 0
    current_fn   = None

    # We'll call this function each time we detect the end of function
    # definition or the start of a new definition.
    def end_current_fn():
        if not capture_mode:
            return
        print_fn_w_docstring('\n'.join(current_fn))

    status_prefix = 'Writing docstrings for each function .. '
    for line_idx, line in enumerate(lines):

        print_status_msg(
                status_prefix + f'{line_idx+1} / {len(lines)}',
                end='\r',
                flush=True
        )

        if m := re.search(r'^(\s*)def ', line):
            end_current_fn()
            capture_mode = True
            indentation  = len(m.group(1))
            current_fn   = [line]  # This will be a list of lines.
        else:
            this_indent = re.search(r'^(\s*)', line)
            this_indent = len(this_indent.group(1))
            if len(line.strip()) > 0 and this_indent <= indentation:
                # We just finished capturing a function definition.
                end_current_fn()
                capture_mode = False
            if capture_mode:
                current_fn.append(line)
            else:
                print_out(line)
    end_current_fn()  # Don't drop a fn defined up to the last line.

    print_status_msg(status_prefix + 'done!' + ' ' * 10)
    print_status_msg(f'\nAll Done! Your updated code is at {output_file_path}')

    if output_file:
        output_file.close()
