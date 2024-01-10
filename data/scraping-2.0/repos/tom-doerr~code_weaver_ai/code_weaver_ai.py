#!/usr/bin/env python3


import openai
import sys
import os
import configparser
import argparse
import subprocess

# Get config dir from environment or default to ~/.config
CONFIG_DIR = os.getenv('XDG_CONFIG_HOME', os.path.expanduser('~/.config'))
API_KEYS_LOCATION = os.path.join(CONFIG_DIR, 'openaiapirc')

# Read the organization_id and secret_key from the ini file ~/.config/openaiapirc
# The format is:
# [openai]
# organization_id=<your organization ID>
# secret_key=<your secret key>

# If you don't see your organization ID in the file you can get it from the
# OpenAI web site: https://openai.com/organizations
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


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='File to process')

initialize_openai_api()

def generate():
    # response = openai.Completion.create(
        # engine="text-davinci-003",
        # prompt='test',
        # stream=True,
    # )

    code_prompt = 'import numpy as np'

    message = {"role": "user", "content": code_prompt}
    messages = [message]

    response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=messages,
            max_tokens=1000,
            stream=True,
            )

    for i, chunk in enumerate(response):
        print(chunk['choices'][0]['delta']['content'], end='')

# def run_program():
    # command = 'docker compose up --build'
    # result = subprocess.run(command, shell=True)
    # return result.returncode


# def run_program():
    # command = 'docker compose up --build'
    # subprocess.run(command, shell=True)

    # # Get the exit code of the service after it has stopped
    # exit_command = 'docker compose ps --services --filter "status=exited" | xargs docker compose ps -q | xargs docker inspect -f "{{.State.ExitCode}}"'
    # exit_code = subprocess.check_output(exit_command, shell=True).decode().strip()

    # return int(exit_code)



def run_program():
    command = 'docker compose up --build'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    stdout = result.stdout
    stderr = result.stderr

    # Extracting the exit code from the output
    # Assuming the format is always "exited with code X", where X is the exit code
    exit_code_line = [line for line in (stdout + '\n' + stderr).split('\n') if "exited with code" in line]
    if exit_code_line:
        exit_code = int(exit_code_line[0].split()[-1])
    else:
        exit_code = result.returncode

    return exit_code, stdout, stderr



if __name__ == "__main__":
    # generate()
    # exit_code = run_program()
    exit_code, stdout, stderr = run_program()
    print(f"Exit code: {exit_code}")

