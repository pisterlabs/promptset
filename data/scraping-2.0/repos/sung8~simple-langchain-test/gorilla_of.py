# Import Chat completion template and set-up variables
import openai
import urllib.parse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

## SETUP
openai.api_key = "EMPTY" # Key is ignored and does not matter
openai.api_base = "http://zanino.millennium.berkeley.edu:8000/v1"
# Alternate mirrors
# openai.api_base = "http://34.132.127.197:8000/v1"


## EXAMPLE FROM GORILLA INTERFACE COLAB
# GitHub: https://github.com/ShishirPatil/gorilla/blob/main/inference/README.md#inference-using-cli
# Colab: https://colab.research.google.com/drive/1DEBPsccVLF_aUnmD0FwPeHFrtdC0QIUP?usp=sharing
# Report issues
def raise_issue(e, model, prompt):
    issue_title = urllib.parse.quote("[bug] Hosted Gorilla: <Issue>")
    issue_body = urllib.parse.quote(f"Exception: {e}\nFailed model: {model}, for prompt: {prompt}")
    issue_url = f"https://github.com/ShishirPatil/gorilla/issues/new?assignees=&labels=hosted-gorilla&projects=&template=hosted-gorilla-.md&title={issue_title}&body={issue_body}"
    print(f"An exception has occurred: {e} \nPlease raise an issue here: {issue_url}")

## Query Gorilla server
def get_gorilla_response(prompt="I would like to translate from English to French.", model="gorilla-7b-hf-v1"):
  try:
    completion = openai.ChatCompletion.create(
      model=model,
      messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content
  except Exception as e:
    raise_issue(e, model, prompt)


# # Gorilla `gorilla-mpt-7b-hf-v1` with code snippets
# # Translation
prompt = "I would like to translate 'I feel very good today.' from English to Chinese."
print(get_gorilla_response(prompt, model="gorilla-7b-hf-v1"))


######## OPEN FUNCTIONS ########

# # open functions 1
# ## DOES NOT WORK
# # source: https://github.com/ShishirPatil/gorilla/tree/main/openfunctions
# def get_gorilla_response2(prompt="Call me an Uber ride type \"Plus\" in Berkeley at zipcode 94704 in 10 minutes", model="gorilla-openfunctions-v0", functions=[]):
#   openai.api_key = "EMPTY"
#   openai.api_base = "http://luigi.millennium.berkeley.edu:8000/v1"
#   try:
#     completion = openai.ChatCompletion.create(
#       model="gorilla-openfunctions-v1",
#       temperature=0.0,
#       messages=[{"role": "user", "content": prompt}],
#       functions=functions,
#     )
#     return completion.choices[0].message.content
#   except Exception as e:
#     print(e, model, prompt)
#
#
# query = "Call me an Uber ride type \"Plus\" in Berkeley at zipcode 94704 in 10 minutes"
# functions = [
#     {
#         "name": "Uber Carpool",
#         "api_name": "uber.ride",
#         "description": "Find suitable ride for customers given the location, type of ride, and the amount of time the customer is willing to wait as parameters",
#         "parameters":  [{"name": "loc", "description": "location of the starting place of the uber ride"}, {"name":"type", "enum": ["plus", "comfort", "black"], "description": "types of uber ride user is ordering"}, {"name": "time", "description": "the amount of time in minutes the customer is willing to wait"}]
#     }
# ]
# get_gorilla_response2(query, functions=functions)

# open functions 2
# def get_prompt(user_query, functions=[]):
#   if len(functions) == 0:
#     return f"USER: <<question>> {user_query}\nASSISTANT: "
#   functions_string = json.dumps(functions)
#   return f"USER: <<question>> {user_query} <<function>> {functions_string}\nASSISTANT: "


# def get_prompt(user_query: str, functions: list = []) -> str:
#     """
#     Generates a conversation prompt based on the user's query and a list of functions.
#
#     Parameters:
#     - user_query (str): The user's query.
#     - functions (list): A list of functions to include in the prompt.
#
#     Returns:
#     - str: The formatted conversation prompt.
#     """
#     if len(functions) == 0:
#         return f"USER: <<question>> {user_query}\nASSISTANT: "
#     functions_string = json.dumps(functions)
#     return f"USER: <<question>> {user_query} <<function>> {functions_string}\nASSISTANT: "
#
# # Device setup
# device : str = "cuda:0" if torch.cuda.is_available() else "cpu"
# torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
#
# # Model and tokenizer setup
# model_id : str = "gorilla-llm/gorilla-openfunctions-v1"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
#
# # Move model to device
# model.to(device)
#
# # Pipeline setup
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=128,
#     batch_size=16,
#     torch_dtype=torch_dtype,
#     device=device,
# )
#
# # Example usage
# query: str = "Call me an Uber ride type \"Plus\" in Berkeley at zipcode 94704 in 10 minutes"
# functions = [
#     {
#         "name": "Uber Carpool",
#         "api_name": "uber.ride",
#         "description": "Find suitable ride for customers given the location, type of ride, and the amount of time the customer is willing to wait as parameters",
#         "parameters":  [
#             {"name": "loc", "description": "Location of the starting place of the Uber ride"},
#             {"name": "type", "enum": ["plus", "comfort", "black"], "description": "Types of Uber ride user is ordering"},
#             {"name": "time", "description": "The amount of time in minutes the customer is willing to wait"}
#         ]
#     }
# ]
#
# # Generate prompt and obtain model output
# prompt = get_prompt(query, functions=functions)
# output = pipe(prompt)
#
# print(output)

# open function 3

# # Example dummy function hard coded to return the same weather
# # In production, this could be your backend API or an external API
# def get_current_weather(location, unit="fahrenheit"):
#     """Get the current weather in a given location"""
#     weather_info = {
#         "location": location,
#         "temperature": "72",
#         "unit": unit,
#         "forecast": ["sunny", "windy"],
#     }
#     return json.dumps(weather_info)
#
# def run_conversation():
#     # Step 1: send the conversation and available functions to GPT
#     messages = [{"role": "user", "content": "What's the weather like in Boston?"}]
#     functions = [
#         {
#             "name": "get_current_weather",
#             "description": "Get the current weather in a given location",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "location": {
#                         "type": "string",
#                         "description": "The city and state, e.g. San Francisco, CA",
#                     },
#                     "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
#                 },
#                 "required": ["location"],
#             },
#         }
#     ]
#     openai.api_key = "EMPTY" # Hosted for free with ❤️ from UC Berkeley
#     openai.api_base = "http://luigi.millennium.berkeley.edu:8000/v1"
#     response = openai.ChatCompletion.create(
#         # model="gpt-3.5-turbo-0613",
#         model='gorilla-openfunctions-v0',
#         messages=messages,
#         functions=functions,
#         function_call="auto",  # auto is default, but we'll be explicit
#     )
#     response_message = response["choices"][0]["message"]
#     print(response_message)
#
#
#
# run_conversation()

