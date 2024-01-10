# Made with <3 by Jacob Terkuc

import openai, os, ast, string, random
import settings, character
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


# This function takes a library 'msg' and calls the OpenAI API to generate a response.
# It returns the response as a string.
def generate(msg):
    print("Generating response...")
    print(msg)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=msg,
    )
    return response


# Function that takes an integer 'num', and an optional string 'desc' and returns a list of 'num' character.
# If 'desc' is not provided, it will default to creating 'num' random character of type 'race'.


# Function that generates a random 8-digit 'id' string and returns it.
def generate_id(length=8):
    letters_and_digits = string.ascii_letters + string.digits
    result_str = ''.join(random.choice(letters_and_digits) for i in range(length))
    return result_str

