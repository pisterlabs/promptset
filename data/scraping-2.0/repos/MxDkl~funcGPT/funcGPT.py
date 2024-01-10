import os
from dotenv import load_dotenv
from openai import OpenAI
from ast import literal_eval

load_dotenv()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

MODEL = "gpt-4-1106-preview"


# A helper function that tries to convert a string to a python type
def simplest_type(s):
    try:
        return literal_eval(s)
    except:
        return s


# A decorator that uses gpt to 'calculate' the output of a function
def funcGPT(function):
    def wrapper(*args, **kwargs):
        system_prompt = "You are a helpful assistant. Given the name of a python function and its arguments, you will return what you think is the expected output of the function. You will not explain your response. You will NEVER preface your response with anything. NEVER end your response with a period. Even if you think the function name is ambiguous."
        prompt = "What is the expected output of the following python function: " + function.__name__ + ", with the following arguments: " + str(args) + " and " + str(kwargs) + "?"
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        response = response.choices[0].message.content
        return simplest_type(response)
    return wrapper