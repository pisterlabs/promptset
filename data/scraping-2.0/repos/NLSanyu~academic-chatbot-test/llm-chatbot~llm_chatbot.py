import os
import openai
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())   # read local .env file

# 'system_message' and 'user_message' are python files containing variables with 
# 1. the message given to the system and 
# 2. the message from the user
# # respectively
import system_message
import user_message


openai.api_key  = os.environ['API_KEY']

def get_completion_from_messages(messages, 
                                 model="gpt-3.5-turbo", 
                                 temperature=0, 
                                 max_tokens=1000):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens,
    )
    return response.choices[0].message["content"]


delimiter = "####"
messages =  [
    {'role':'system', 
    'content': system_message.message},
    {'role':'user', 
    'content': f"{delimiter}{user_message.message}{delimiter}"}
] 
response = get_completion_from_messages(messages)

response_file = os.path.join(os.getcwd(), "llm_response.txt")
with open(response_file, "a") as f:
    f.write(f"\n\n{response}")

