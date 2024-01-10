import openai
import json

openai.api_key_path = './openapi_key.txt' # Your API key should be here

story = "As a repository manager, I want to know all the collections and objects in the DAMS for which I have custodial responsibility."

conversation = list()

schema = { 
      "name": "record_elements",
      "description": "Record the elements extracted from a story",
      "parameters": {
        "type": "object",
        "properties": {
          "personas": {
            "type": "array",
            "description": "The list of personas extracted from the story",
            "items": { "type": "string" }
          }
        }
      }
    }

conversation.append(
    {'role': 'system', 'content': 'You are a requirements engineering assistant. You will be provided by the user a user story, and your task is to extract element from these models and call provided functions to record your findings.'})
conversation.append(
    {'role': 'system', 'content': 'You are only allowed to call the provided function in your answer'})
conversation.append({'role': 'user', 'content': "Here is the story you have to process:\n"+story})

response = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo-0613",
    functions = [ schema ],
    messages = conversation,
    temperature=0.0)

print(json.dumps(response, indent=2))

