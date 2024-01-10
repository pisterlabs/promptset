#OpenAI python official github => https://github.com/openai/openai-python
#https://python.langchain.com/docs/integrations/llms/openai
import os
from openai import OpenAI
# based on openAI version is higher than 1.0.0[pip install openai==1.6.1]
# python3 localLLM/testOpenAI.py
from langchain.globals import set_debug
from langchain.globals import set_verbose
from getpass import getpass

#for debuging refer:https://python.langchain.com/docs/guides/debugging
set_debug(True)
set_verbose(True)
print("==============================================Completed the setup env=========================")
#==============================================Completed the setup env=========================


#==============================================exact logic code with openAI version higher 1.0.0=========================
#Initialize LLM language model
# TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(api_base="http://192.168.0.232:1234/v1")'
base_url = os.environ.get("TEST_API_BASE_URL", "http://192.168.0.232:1234/v1")     #This is working as well for LM Studio server
#base_url = os.environ.get("TEST_API_BASE_URL", "http://192.168.0.232:3000/v1")    #This code not work for openLLM server
api_key = os.environ.get("OPENAI_API_KEY", "xxxxxxxxx")   # even your local don't need to do the authorization, but you need to fill something, otherwise will be get exception.
api_key = "xxxx"
openAI_client = OpenAI(
    base_url = base_url,
    # This is the default and can be omitted
    api_key=api_key,
    _strict_response_validation=False,
)

# messages=[
#     {"role": "system", "content": "Always answer in rhymes."},
#     {"role": "user", "content": "Introduce yourself."},
#     {
#             "role": "user",
#             "content": "Say this is a test",
#     }
#   ]
messages = [{"role": "user", "content": "hi"}]

#model = "/Users/sl6723/.cache/lm-studio/models/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q8_0.gguf"
model = "facebook--opt-1.3b"
chat_completion=openAI_client.chat.completions.create(
    messages=messages, 
    model=model, # this field is currently unused
    stream=False,
)
print(chat_completion.choices[0].message.model_dump())
#print(chat_completion.choices[0].message)
#print(chat_completion)

