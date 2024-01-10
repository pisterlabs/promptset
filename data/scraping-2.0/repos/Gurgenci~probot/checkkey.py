import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

OpenAI_KEY_notset="\n\nERROR *** OPENAI_API_KEY is not set. If you have an OpenAI key, let this \
program know about it:\n\1. Create the file .env in the project folder\n\
2. Enter the line OPENAI_API_KEY='my-api-key-here' in the .env file\n\
3. Add `.env` to the .gitignore file so that the key is not shared with others"

def checkkey():
    if openai.api_key == None:
        print(OpenAI_KEY_notset)
        exit(1)
    else:
        print("OpenAI API key is set")
        print("OpenAI API key is: ", openai.api_key)
        print("OpenAI API key type is: ", type(openai.api_key))
        print("OpenAI API key length is: ", len(openai.api_key))
        print("OpenAI API key is: ", openai.api_key[0:5], "...")
        print("OpenAI API key is: ", openai.api_key[-5:])
        return openai.api_key

if __name__ == "__main__":
    checkkey()
