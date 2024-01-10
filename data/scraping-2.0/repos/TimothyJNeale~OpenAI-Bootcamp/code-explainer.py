import openai
from dotenv import load_dotenv
import os

import inspect

# load environment variables from .env file
load_dotenv()

############################# Helper Functions ###############################

# Use chat completion
def get_chat_completion(prompt, model="gpt-3.5-turbo", temperature=0):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message["content"]

# Instruction completion
def get_completion(prompt, model="gpt-3.5-turbo-instruct", temperature=0, max_tokens=300, stop="\"\"\""):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop)

    return response.choices[0].text

def docstring_prompt(code):
    prompt = f"{code}\n  # A high quality python docstring for the above python function:\n  \"\"\"\n  "
    return prompt   

def merge_docstring(code, docstring):
    # First line of the function
    # Inset first 3 """" and newline
    # Insert the docstring (ends in """") and newline
    # Insert the rest of the function
    code = code.split("\n")
    code = code[:1] + ["\"\"\"\n" + docstring.strip()] + ["\n\"\"\""] + code[1:] 
    return "\n".join(code)

############################### Authenticate #################################

# Authenticate with OpenAI                             
api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = api_key


############################### Main Program #################################

def hello(name):
    print(f"Hello {name}!")

source = inspect.getsource(hello)
prompt = docstring_prompt(source)
docstring = get_completion(prompt)
#print(docstring)
new_source = merge_docstring(source, docstring)
print(new_source)

