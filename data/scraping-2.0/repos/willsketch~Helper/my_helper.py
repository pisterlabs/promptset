import os
import openai
import argparse
from helper.main import Helper

def check_api_key():
    """this function checks if openai api key has been set"""

    if openai.api_key is  None:

        print('you do not have an api key yet\n Please follow instructions on the README on how to get the api key')
        return 'API_KEY is None'


def cli():
    """
    this function creates the cli tool using argparse
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")

    check_api_key()

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', help='What do you need help with', default='')
    query= parser.parse_args().prompt
    response = Helper(query).run_engine()
    print(response.choices[0].text)

if __name__ == '__main__':
    cli()
