import os
import openai
from sys import exit
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

OPENAI_KEY = os.environ.get("OPENAI_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL")

if OPENAI_KEY and OPENAI_MODEL:
    try:
        openai.api_key = OPENAI_KEY
    except:
        print("NO :(")
    print("Keys loaded successfully!")
else:
    print("Error in setting up the environment")
    exit()

def main():
    chat_completion = openai.ChatCompletion.create(model=OPENAI_MODEL, messages=[{"role": "user", "content": "Hello world"}])
    print(chat_completion)

if __name__ == "__main__":
    main()