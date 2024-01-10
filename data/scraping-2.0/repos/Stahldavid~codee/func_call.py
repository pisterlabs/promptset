

# import json
# import openai
# import requests
# from tenacity import retry, wait_random_exponential, stop_after_attempt
# from termcolor import colored

# GPT_MODEL = "gpt-3.5-turbo-0613"

# @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
# def chat_completion_request(messages, functions=None, function_call=None, model=GPT_MODEL):
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": "Bearer " + openai.api_key,
#     }
#     json_data = {"model": model, "messages": messages}
#     if functions is not None:
#         json_data.update({"functions": functions})
#     if function_call is not None:
#         json_data.update({"function_call": function_call})
#     try:
#         response = requests.post(
#             "https://api.openai.com/v1/chat/completions",
#             headers=headers,
#             json=json_data,
#         )
#         return response
#     except Exception as e:
#         print("Unable to generate ChatCompletion response")
#         print(f"Exception: {e}")
#         return e

# def pretty_print_conversation(messages):
#     role_to_color = {
#         "system": "red",
#         "user": "green",
#         "assistant": "blue",
#         "function": "magenta",
#     }
#     formatted_messages = []
#     for message in messages:
#         if message["role"] == "system":
#             formatted_messages.append(f"system: {message['content']}\n")
#         elif message["role"] == "user":
#             formatted_messages.append(f"user: {message['content']}\n")
#         elif message["role"] == "assistant" and message.get("function_call"):
#             formatted_messages.append(f"assistant: {message['function_call']}\n")
#         elif message["role"] == "assistant" and not message.get("function_call"):
#             formatted_messages.append(f"assistant: {message['content']}\n")
#         elif message["role"] == "function":
#             formatted_messages.append(f"function ({message['name']}): {message['content']}\n")
#     for formatted_message in formatted_messages:
#         print(
#             colored(
#                 formatted_message,
#                 role_to_color[messages[formatted_messages.index(formatted_message)]["role"]],
#             )
#         )


# functions = [
#     {
#         "name": "get_current_weather",
#         "description": "Get the current weather",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "location": {
#                     "type": "string",
#                     "description": "The city and state, e.g. San Francisco, CA",
#                 },
#                 "format": {
#                     "type": "string",
#                     "enum": ["celsius", "fahrenheit"],
#                     "description": "The temperature unit to use. Infer this from the users location.",
#                 },
#             },
#             "required": ["location", "format"],
#         },
#     },
#     {
#         "name": "get_n_day_weather_forecast",
#         "description": "Get an N-day weather forecast",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "location": {
#                     "type": "string",
#                     "description": "The city and state, e.g. San Francisco, CA",
#                 },
#                 "format": {
#                     "type": "string",
#                     "enum": ["celsius", "fahrenheit"],
#                     "description": "The temperature unit to use. Infer this from the users location.",
#                 },
#                 "num_days": {
#                     "type": "integer",
#                     "description": "The number of days to forecast",
#                 }
#             },
#             "required": ["location", "format", "num_days"]
#         },
#     },
# ]



# messages = []
# messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
# messages.append({"role": "user", "content": "What's the weather like today"})

# chat_response = chat_completion_request(
#     messages, functions=functions
# )

# assistant_message = chat_response.json()["choices"][0]["message"]
# messages.append(assistant_message)




# def execute_function_call(message):
#     if message["function_call"]["name"] == "get_current_weather":
#         # Execute get_current_weather function with the generated arguments
#         location = eval(message["function_call"]["arguments"])["location"]
#         format = eval(message["function_call"]["arguments"])["format"]
#         output = get_current_weather(location, format)
#     elif message["function_call"]["name"] == "get_n_day_weather_forecast":
#         # Execute get_n_day_weather_forecast function with the generated arguments
#         location = eval(message["function_call"]["arguments"])["location"]
#         format = eval(message["function_call"]["arguments"])["format"]
#         num_days = eval(message["function_call"]["arguments"])["num_days"]
#         output = get_n_day_weather_forecast(location, format, num_days)
#     else:
#         output = "Invalid function call"
#     return output

# def get_current_weather(location, format):
#     # Implement get_current_weather function logic here


# def get_n_day_weather_forecast(location, format, num_days):
#     # Implement get_n_day_weather_forecast function logic here




import openai
import pinecone
import json
from dotenv import load_dotenv
import os
from datetime import datetime


# Load environment variables from .env file
load_dotenv()

# Access the API key from the environment variable
api_key = os.getenv('PINECONE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')


# OpenAI setup
openai.api_key = openai_api_key
api_key = api_key

# Pinecone setup
pinecone.init(api_key=api_key, environment='us-east1-gcp')
index = pinecone.Index("langchain-chat")

# Define your function to transform the query into a vector using OpenAI's text embeddings
def transform_query_to_vector(query):
    """Transform a text query into a vector using OpenAI's text embeddings."""
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=query
    )
    return response['data'][0]['embedding']

# Define your function to query the Pinecone vector store
def get_robotics_knowledge(query):
    """Query the Pinecone vector store using a query transformed into a vector. It is used to retrieve knowledge on robotics."""
    # Transform your query into a vector
    query_vector = transform_query_to_vector(query)
    
    # Query Pinecone
    result = index.query(
        vector=query_vector,
        top_k=2,
        include_metadata=True
    )
   
    contexts = [
        x['metadata']['text'] for x in result['matches']
    ]
    print(contexts)
  
    unified_string = ' '.join(contexts)
    return unified_string
      # append contexts until hitting limit


completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=[{"role": "user", "content": "What's impedance control?"}],
    functions=[
    {
        "name": "get_robotics_knowledge",
        "description": "Get retrive knowledge based on a query",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to be retrived",
                },
               
            },
            "required": ["query"],
        },
    }
],
function_call="auto",
)

reply_content = completion.choices[0]
reply_content


if completion['choices'][0]["finish_reason"] == "function_call":
    print("We should call a function!")
    name = completion['choices'][0]['message']['function_call']['name']
    args = json.loads(completion['choices'][0]['message']['function_call']['arguments'])
    name, args
    print("Calling function: ", name)
    print("With arguments: ", args)

mess= get_robotics_knowledge(args['query'])
print(mess)
# res = get_robotics_knowledge("What is Impedance control")
# print(res)