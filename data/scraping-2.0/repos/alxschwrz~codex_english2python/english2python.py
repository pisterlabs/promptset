import os
import configparser
import sys
import contextlib

import openai

MAX_TOKEN_DEFAULT = 128
TEMPERATURE = 0.0

API_KEYS_LOCATION = "./config"
STREAM = True


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


def create_input_prompt(englishTextIn=""):
    prompt = "The following bot transforms natural language to Python code. \n" + \
             "Input: {}\n".format(englishTextIn) + \
             "Python: \n '''python"
    return prompt


def read_from_command_line():
    englishTextIn = input("Input your request in natural language: ")
    return englishTextIn


def generate_completion(input_prompt, num_tokens=MAX_TOKEN_DEFAULT):
    stop_string = "'''\n"
    response = openai.Completion.create(engine='code-davinci-002', prompt=input_prompt, temperature=TEMPERATURE,
                                        max_tokens=num_tokens, stream=STREAM, stop=stop_string,
                                        top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0)
    return response


def get_generated_response(response):
    generatedCode = "## Python code generated from plain english: \n"
    while True:
        nextResponse = next(response)
        completion = nextResponse['choices'][0]['text']
        generatedCode = generatedCode + completion
        if nextResponse['choices'][0]['finish_reason'] is not None:
            break
    return generatedCode


def run():
    loop = True
    while loop:
        englishText = read_from_command_line()
        if "exit()" in englishText:
            loop = False
            break
        prompt = create_input_prompt(englishText)
        response = generate_completion(prompt)
        generatedCode = get_generated_response(response)
        print(generatedCode)


if __name__ == "__main__":
    initialize_openai_api()
    run()
