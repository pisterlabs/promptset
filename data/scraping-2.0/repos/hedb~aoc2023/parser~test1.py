import json

from openai import OpenAI
from openai.types.chat import ChatCompletion


#https://platform.openai.com/docs/api-reference/chat/create
def openai_chat_with_tool():
    client = OpenAI()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            }
        }
    ]
    messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    print(completion)

def openai_chat(prompt):
    client = OpenAI()

    messages = [{"role": "user", "content": prompt}]

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    return completion


if __name__ == '__main__':
    # openai_chat_with_tool()

    prompt = """
I'd like you to parse the following input text into a valid JSON.
We will do it in a few steps process.

First, I'll supply the input and I'd like you to read it and attempt to analyze yourself the intended JSONs.
I'll later supply you with modification requests.

I'll provide a few lines but please return just the first two ones.
provide them as a JSON per line.

the Input:
Game 1: 3 blue, 4 red; 1 red, 2 green, 6 blue; 2 green
Game 2: 1 blue, 2 green; 3 green, 4 blue, 1 red; 1 green, 1 blue
Game 3: 8 green, 6 blue, 20 red; 5 blue, 4 red, 13 green; 5 green, 1 red
Game 4: 1 green, 3 red, 6 blue; 3 green, 6 red; 3 green, 15 blue, 14 red
Game 5: 6 red, 1 blue, 3 green; 2 blue, 1 red, 2 green
"""


    ret:ChatCompletion = openai_chat(prompt)
    res_str = ret.choices[0].message.content
    try :
        res = json.loads(res_str)
        # pretty print the json
        print(json.dumps(res, indent=4, sort_keys=True))
    except:
        print("failed to parse the result as json")
        print(res_str)


    
    
