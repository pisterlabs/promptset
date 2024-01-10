#OpenAI python official github => https://github.com/openai/openai-python

import os
import openai
# based on openAI version is less than 1.0.0[pip install openai==0.28]
# python3 localLLM/testOldOpenAI.py
# Do this so we can see exactly what's going on under the hood
from langchain.globals import set_debug
from langchain.globals import set_verbose
from getpass import getpass

#for debuging refer:https://python.langchain.com/docs/guides/debugging
set_debug(True)
set_verbose(True)
print("==============================================Completed the setup env=========================")
#==============================================Completed the setup env=========================


#==============================================exact logic code with openAI version less 1.0.0=========================
#Initialize LLM language model
## Local own LLM API refer:https://python.langchain.com/docs/integrations/llms/openllm
#server_url = "http://localhost:3000"  # Replace with remote host if you are running on a remote server
# server_url = "http://192.168.0.232:1234"  # Replace with remote host if you are running on a remote server
# llm = OpenLLM(server_url=server_url)
# TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(api_base="http://192.168.0.232:1234/v1")'
openai.api_base = "http://192.168.0.232:1234/v1" # point to the local server
openai.api_key = "" # no need for an API key

messages=[
    {"role": "system", "content": "Always answer in rhymes."},
    {"role": "user", "content": "Introduce yourself."}
  ]
completion = openai.ChatCompletion.create(
  model="local-model", # this field is currently unused
  messages=messages,
)
print(completion.choices[0].message)
