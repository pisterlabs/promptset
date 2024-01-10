#!/usr/bin/env python3

'''
This script reads program files and concatenates the beginning of
all files to create a input prompt which is then fed to OpenAI
Codex to generate a README.
'''
import sys
import traceback

# Check if the openai module is installed.
try:
    import openai
except ImportError:
    print('openai module not found. Try running "pip3 install openai"')
    sys.exit(1)

import os
import argparse
import configparser

FILES_NOT_TO_INCLUDE = ['LICENSE', 'README.md']
STREAM = True
cur_dir_not_full_path = os.getcwd().split('/')[-1]
README_START =  f'# {cur_dir_not_full_path}\n## What is it?\n'

# Get config dir from environment or default to ~/.config
CONFIG_DIR = os.getenv('XDG_CONFIG_HOME', os.path.expanduser('~/.config'))
API_KEYS_LOCATION = os.path.join(CONFIG_DIR, 'openaiapirc')

def create_template_ini_file():
    """
    If the ini file does not exist create it and add the organization_id and
    secret_key
    """
    if not os.path.isfile(API_KEYS_LOCATION):
        with open(API_KEYS_LOCATION, 'w') as f:
            f.write('[openai]\n')
            f.write('organization_id=\n')
            f.write('secret_key=\n')

        print('OpenAI API config file created at {}'.format(API_KEYS_LOCATION))
        print('Please edit it and add your organization ID and secret key')
        print('If you do not yet have an organization ID and secret key, you\n'
               'need to register for OpenAI Codex: \n'
                'https://openai.com/blog/openai-codex/')
        sys.exit(1)

def initialize_openai_api():
    """
    Initialize the OpenAI API
    """
    # Check if file at API_KEYS_LOCATION exists
    create_template_ini_file()
    config = configparser.ConfigParser()
    config.read(API_KEYS_LOCATION)

    openai.organization_id = config['openai']['organization_id'].strip('"').strip("'")
    openai.api_key = config['openai']['secret_key'].strip('"').strip("'")

def create_input_prompt(max_length=2000):
    input_prompt = ''
    files_sorted_by_mod_date = []

    # Esplora ricorsivamente tutte le directory e sottodirectory
    for root, dirs, files in os.walk('.'):
        for filename in files:
            if filename == 'README.md':  # Aggiungi questo controllo per escludere il file README.md
                continue

            file_path = os.path.join(root, filename)
            files_sorted_by_mod_date.append(file_path)

    # Ordina i file in base alla data di modifica
    files_sorted_by_mod_date.sort(key=os.path.getmtime, reverse=True)

    prompt_length = 0
    for filename in files_sorted_by_mod_date:
        if prompt_length >= max_length:
            break

        # Aggiungi il controllo per verificare se l'elemento corrente è un file
        if not os.path.isfile(filename):
            continue

        with open(filename, 'r', errors='ignore') as f:  # Aggiungi l'opzione errors='ignore' qui
            file_content = f.read()
            content_to_add = f'\n===================\n# {filename}:\n{file_content}\n'
            if prompt_length + len(content_to_add) <= max_length:
                input_prompt += content_to_add
                prompt_length += len(content_to_add)

    input_prompt += '\n\n===================\n# ' + 'README.md:' + '\n'
    input_prompt += README_START

    return input_prompt


def generate_completion(input_prompt, num_tokens):
    response = openai.Completion.create(engine='text-davinci-001', prompt=input_prompt, temperature=0.5, max_tokens=num_tokens, stop='===================\n')
    return response


def generate_completion_chatgpt(input_prompt, num_tokens):
    messages = [
            {'role': 'user', 
            'content': input_prompt}
            ]

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        max_tokens=num_tokens,
        temperature=0.8,
        messages=messages,
    )

    return response

def clear_screen_and_display_generated_readme(response):
    # Clear screen.
    os.system('cls' if os.name == 'nt' else 'clear')
    generated_readme = ''
    print(README_START)

    completion = response['choices'][0]['text']
    print(completion)
    generated_readme = README_START + completion

    return generated_readme


def clear_screen_and_display_generated_readme_chatgpt(response):
    # Clear screen.
    os.system('cls' if os.name == 'nt' else 'clear')
    generated_readme = ''
    print(README_START)

    response_text = response["choices"][0]["message"]['content']
    print(response_text)
    return response_text

def save_readme(readme_text):
    '''
    Saves the readme.
    If a readme already exists ask the user whether he wants
    to overwrite it.
    '''
    if os.path.isfile('README.md'):
        answer = input('A README.md already exists. Do you want to overwrite it? [y/N] ')
        if answer == '' or answer == 'n' or answer == 'N':
            print('\nThe README was not saved.')
            return

    with open('README.md', 'w') as f:
        f.write(readme_text)

    print('\nREADME.md saved.')

def generate_until_accepted(input_prompt, num_tokens):
    '''
    Generate new readmes and ask the user if he wants to save the generated
    readme.
    '''
    while True:
        try:
            if args.chatgpt:
                response = generate_completion_chatgpt(input_prompt, num_tokens)
                generated_readme = clear_screen_and_display_generated_readme_chatgpt(response)
            else:
                response = generate_completion(input_prompt, num_tokens)
                generated_readme = clear_screen_and_display_generated_readme(response)
        except Exception as e:
            print("An error occurred during the generation:")
            print(traceback.format_exc())
            continue

        # Ask the user if he wants to save the generated readme.
        answer = input("\n\nDo you want to save the generated README? [y/N] ")
        if answer == '' or answer == 'n' or answer == 'N':
            print('\nThe generated README is not saved.')
            continue
        elif answer == 'y' or answer == 'Y':
            save_readme(generated_readme)

        answer = input("\n\nDo you want to generate another README? [Y/n] ")
        if answer == '' or answer == 'y' or answer == 'Y':
            continue
        break


def get_args():
    # Get the number of tokens as positional argument.
    parser = argparse.ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--chatgpt", action='store_true')
    parser.add_argument("--num_versions", type=int, default=1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    initialize_openai_api()
    input_prompt = create_input_prompt()
    for _ in range(args.num_versions):
        generate_until_accepted(input_prompt, args.tokens)
