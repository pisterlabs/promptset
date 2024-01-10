from secret_key import openai_key
import openai
import json
openai.api_key = openai_key

## Now if we want to know about current boston weather . We wont be able to get that data 
## So we will create a dummy function.Which will be eventually connected to Database 

def get_current_weather(location,unit='fahrenheit'):
    ## We can also use API/Database to get the weather of location
    weather_info = {
        "location":location,
        "temperature":"72",
        "unit" : unit,
        "forecast":["sunny","windy"]
    }
    return json.dumps(weather_info)

## What function OpenAI API can call .If it doesnt find any information in its own knowledge
functions =[
        {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                            },
                "unit": {
                    "type": "string", 
                    "enum": ["celsius", "fahrenheit"]
                    },
                },
                "required": ["location"],
            },
        }        
    ]

messages = [
        {"role":"user","content":"What is the weather in Boston"}
    ]

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages = messages,
    ## functions will be used when chatgpt cannot find answer in their own knowledge
    ## Because of reasoning capability . It can map question to thier corresponding function
    ## From Question - Reasoning will automically select right entities. Like Boston is city . So it will be our location in above function
    functions = functions,
    function_call = "auto"

)

'''
You will get reponse in following mannner

{
  "role": "assistant",
  "content": null,
  "function_call": {
    "name": "get_current_weather",
    "arguments": "{\n  \"location\": \"Boston\"\n}"
  }
}

'''

## So we will extract function_call['name'] and function_call['arguments']
response_message = response['choices'][0]['message']
function_name = response_message['function_call']['name']
function_args = json.loads(response_message['function_call']['arguments'])

## So from response message first we are identifying which function we will call based on question
## Then we will choose it from available functions where we have mapped all of our functions

available_functions = {
    'get_current_weather':get_current_weather,   
}

function_response = available_functions[function_name](location=function_args.get('loaction'))
print(function_response)

## Above will print following 
## {"location": null, "temperature": "72", "unit": "fahrenheit", "forecast": ["sunny", "windy"]} 

## Above output is not in human readable format . So to make above json output in human readable format . We will again use LLM
## We have created messages list . We will append few more prompts over there .

messages.append(response_message) ## Extend conversation with assistant reply 
messages.append(
    {
        "role":"function",
        "name":function_name, ## We have already extracted function name from response message 
        "content":function_response
    }
)
## We will call LLM Again

second_response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo-0613',
    messages = messages
) # get a response from GPT where it can see function response

print(second_response)

print('\nOutput')
print(second_response['choices'][0]['message']['content'])



