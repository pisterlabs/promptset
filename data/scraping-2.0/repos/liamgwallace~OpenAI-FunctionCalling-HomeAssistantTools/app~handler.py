import json
import os
import pprint
import openai
from dotenv import find_dotenv, load_dotenv
from datetime import datetime
from dateutil.tz import tzlocal
#gpt-3.5-turbo-0613
#gpt-4-0613

class OpenAIHandler:
    def __init__(self, api_functions, function_definitions, system_role="", model="gpt-3.5-turbo-0613", temperature=0):
        load_dotenv(find_dotenv())
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        if openai.api_key is None:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")

        self.api_functions = api_functions
        self.function_definitions = function_definitions
        self.model = model
        self.temperature = temperature
        self.system_role = """
You are professional level decision making assistant.
You have access to a variety of functions that can help the user. 
You will do 3 things.
 1. create a step by step plan and output this as a system message
 2. pick the next action on the plan and use the functions at your disposal to output the function and details as a system function
 3. If the plan is complete and you have enough to respond to the user then ignore 1 and 2 and output the reponse to the user in a system message
 
 
1. Planning
Let's devise a step-by-step plan to address the user query .
When outlining each step, provide a succinct description.
If you can foresee the need for any functions or parameters, mention them explicitly.
If unsure, use placeholders.
The sole output should be the plan, nothing more.

First, list all the tools you have at your disposal.
Then list ones that might help answer this question.
There may be many steps required, so start by searching for the information you need to make any function requests.
Don't assume anything, you dont know anything other than the information you have been told.

Each step should be in the following format:

{
  'Subtask': '<details of the problem and goal>',
  'Reasoning': '<small step to solve the problem>',
  'Function': '<list the function call that might help this step>',
  'Parameters': '<additional parameters to pass to the function>'
}

Represent the plan in the following format:

{
  'Complete_steps': [<list of steps>],
  'Current_step': [<list of steps>],
  'Next_steps': [<list of steps>]
}

2. Use a function
You must choose the next function to call in order to help the user where possible. 
Refer to the plan and any results from previous function calls you have made to select the function and populate the correct fields
Make your best attempt to call a function where possible. Only respond to the user if you cannot solve the task or have solved the task.
Only use the functions you have been provided with. Do not guess at the data needed for a function, try and see if you can search for what is needed using a function at your disposal.

3. Task complete :)
DO NOT CALL A FUNCTION. Just return a response to the user.
"""

    def generate_AI_message(role, content):
        return f'[{{"role": "{role}", "content": {content}}}]'

    def send_message(self, messages):
        # print(f"\n\n############   messages to llm   ##############")
        # pprint.pprint(messages)
        # pprint.pprint(self.function_definitions)
        # print(f"############   end   ##############\n")
        response = openai.ChatCompletion.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
            functions=self.function_definitions,
        )
        print(f"\n\n############   message response from llm   ##############")
        pprint.pprint(response)
        # print(response["choices"][0]["message"]["content"])
        print(f"############   end   ##############\n")
        response = response["choices"][0]["message"]
        return response

    def process_function_call(self, response):
        function_name = None
        function_args = None
        planning = None
        result = None
     
        try:
            if 'function_call' in response:
                # print("DEBUG:function call found")
                function_call = response['function_call']
                function_name = function_call.get('name')                
                # Ensure that arguments is a dictionary
                function_args_dict = function_call.get('arguments', {})
                if isinstance(function_args_dict, str):
                    function_args_dict = json.loads(function_args_dict)
                #Strip the planning off of the function call resoponse
                # planning = function_args_dict.pop('planning', None)
                # print(f"\n\n############   planning   ##############")
                # pprint.pprint(planning)
                # print(f"############   end   ##############\n")
                function_args_json = json.dumps(function_args_dict)
                function_args = json.loads(function_args_json)
                print(f"\n\n############   Found function   ##############")
                print(f"function_name: {function_name}")
                print(f"Arguments: {function_args}")
                print(f"############   end   ##############\n")

        except json.JSONDecodeError as e:
            result = f"Error decoding JSON: {e}"
            print(result)

        except Exception as e:
            result = f"Unexpected error: {e}"
            print(result)

        if function_name and not result:
            api_function = self.api_functions.get(function_name)
            if api_function:
                try:
                    result = str(api_function(**function_args))
                except Exception as e:
                    result = f"Error calling function '{function_name}': {e}"
                    print(result)
            else:
                result = f"Function '{function_name}' not found"
                print(result)

        return function_args, function_name, result

    def send_response(self, query):
        system_prompt = """
Let's devise a step-by-step plan to address the user query .
When outlining each step, provide a succinct description.
If you can foresee the need for any functions or parameters, mention them explicitly.
If unsure, use placeholders.
The sole output should be the plan, nothing more.

First, list all the tools you have at your disposal.
Then list ones that might help answer this question.
There may be many steps required, so start by searching for the information you need to make any function requests.
Don't assume anything, you dont know anything other than the information you have been told.

Each step should be in the following format:

{
  'Subtask': '<details of the problem and goal>',
  'Reasoning': '<small step to solve the problem>',
  'Function': '<list the function call that might help this step>',
  'Parameters': '<additional parameters to pass to the function>'
}

Represent the plan in the following format:

{
  'Complete_steps': [],
  'Current_step': [],
  'Next_steps': [<list all the steps here>]
}

DO NOT CALL A FUNCTION. Just return a response to the user.
"""
        example_prompt = """
%%%Example Question:
{find a good day this week to walk the dog and add it to my todos.}

%%%Example Answer
{
  Available functions:[list],
  Functions I may need for this task:[list],
  Plan:{
  "Complete_steps": [],
  "Current_step": [],
  "Next_steps": [
    {
      "Subtask": "Check the weather forecast for the current week",
      "Reasoning": "First I need to find a good day to walk the dog, we need to retrieve the weather forecast for the current week.",
      "Function": "pw_get_weather_forecast",
      "Parameters": {
        "location": "Cranleigh, Surrey",
        "forecast_type": ["daily"]
      }
    },
    {
      "Subtask": "Store the selected day in the todos",
      "Reasoning": "From the returned forecast I need to select the best day. e.g. avoid rain and wind. Si I can create a new todo item with the task 'Walk the dog' and the selected day as the due date.",
      "Function": "create_todo",
      "Parameters": {
        "todo": {
          "task": "walk dog on <best day and date>"
        }
      }
    },
    {
      "Subtask": "Return short output to user with my choice and basic reason. e.g. "I have checked this weeks forecast and <day&date> looks the nicest day for a walk. I have added it to your todo list.",
      "Reasoning": "I now have all the information needed to respond to the user, no further function calls are needed.",
      "Function": "none",
    }
  ]
}
"""

        print(f"query: {query}")
        user = "This house belongs to Liam Wallace and Jenny Wallace"
        location = "Your location is Cranleigh, Surrey, UK"
        formatted_now = f" the current time is {datetime.now(tzlocal()).strftime('%Y-%m-%dT%H:%M:%S%z')}"
        #create a detailed plan to solve the user request
        messages_system = [{"role": "system", "content": f"{system_prompt}. {formatted_now}. {location}. {user}\n{example_prompt}"}]        
        messages_user = [{"role": "user", "content": query}]
        messages_plan_create = messages_system + messages_user
        messages_first_plan = [self.send_message(messages_plan_create)]
        messages_first_plan[0]['content'] = f"To answer the users query, this is the original plan:\n{messages_first_plan[0]['content']}"
        
        #setup the messages for the loop
        messages_curr_plan = []
        messages_functions=[]
        messages_system = [{"role": "system", "content": f"{self.system_role}. {formatted_now}. {location}. {user}\n{example_prompt}"}]
        messages_user = [{"role": "user", "content": query}]
        while True:
            messages = messages_system + messages_user + messages_first_plan + messages_curr_plan + messages_functions
            response = self.send_message(messages)
            function_args, function_name, result = self.process_function_call(response)
            if result:
                function_message = {
                    "role": "function","name": function_name if function_name else "unknown",
                    "content": f"function args: {json.dumps(function_args)}\nresponse:\n{result}",
                }
                messages_functions.append(function_message)
                messages_curr_plan = [{"role": "assistant", "content": f"To answer the users request, I have made the following plan. It needs to be updated based on the results from the function and used if I make another function call in the 'planning' field.\nPlan: '{response['content']}'"}]
            else:
                return response["content"]