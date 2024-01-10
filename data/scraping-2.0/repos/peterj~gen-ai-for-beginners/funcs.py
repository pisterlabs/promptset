import openai
from os import environ
from dotenv import load_dotenv
import json
import requests

load_dotenv()
openai.key = environ.get("OPENAI_API_KEY")

def search_courses(role, product, level):
  url = "https://learn.microsoft.com/api/catalog/"
  params = {
     "role": role,
     "product": product,
     "level": level
  }
  response = requests.get(url, params=params)
  modules = response.json()["modules"]
  results = []
  for module in modules[:5]:
     title = module["title"]
     url = module["url"]
     results.append({"title": title, "url": url})
  return str(results)

tools = [
   {
     "type": "function",
     "function": {
      "name":"search_courses",
      "description":"Retrieves courses from the search index based on the parameters provided",
      "parameters":{
         "type":"object",
         "properties":{
            "role":{
               "type":"string",
               "description":"The role of the learner (i.e. developer, data scientist, student, etc.)"
            },
            "product":{
               "type":"string",
               "description":"The product that the lesson is covering (i.e. Azure, Power BI, etc.)"
            },
            "level":{
               "type":"string",
               "description":"The level of experience the learner has prior to taking the course (i.e. beginner, intermediate, advanced)"
            }
         },
         "required":[
            "role"
         ]
      }
    }
   }
]

messages =[{'role': 'user', 'content': "find me a good course for a advanced student to learn Azure."}]

response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        tools=tools,
        tool_choice="auto",
)

msg=response.choices[0].message

if msg.tool_calls:
  # Parsing the structured response we got from chatgpt
  print("recommended tool call:")
  function = msg.tool_calls[0].function
  print(function)
  
  function_name = function.name

  available_functions = {
    "search_courses": search_courses
  }
  function_to_call = available_functions[function_name]
  function_args = json.loads(function.arguments)

  # Actually calling our python function (no LLM involved here)
  function_response = function_to_call(**function_args)

  print("Output of function call:")
  print(function_response)
  print(type(function_response))

  messages.append(
      {
          "role": "function",
          "name": function_name,
          "content":function_response,
      }
  )

  second_response = openai.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=messages,
          tools=tools,
          temperature=0.0,
  )
  
  print("Second response:")
  print(second_response.choices[0].message.content)
