from dotenv import load_dotenv
import os
import openai
import json

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


# openai.organization = os.getenv("OPENAI_ORG")


def query_openai_chat_completion(messages, functions=None, function_call="auto"):
    if functions is None:
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=messages, temperature=0.7)
    else:
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=messages, temperature=0.7,
                                                  functions=functions, function_call=function_call)
    reply = completion.choices[0].message
    return reply


class Agent:
    def __init__(self, schema, query):
        self.json_format = schema
        self.query = query

    def run(self):
        sys_prompt = f"""
Reply using the specified json format only 
```json
{self.json_format}
```
            """
        user_prompt = f"{self.query}"
        print(sys_prompt)
        print(user_prompt)
        messages = [
            {
                "role": "system", "content": sys_prompt
            },
            {
                "role": "user", "content": user_prompt
            },
        ]
        reply = query_openai_chat_completion(messages)
        return reply.content


if __name__ == "__main__":
    schema = {
        "dependencies": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    }
    query = "In a shallow wide bowl, whisk together the milk, cornstarch, ground flaxseeds, baking powder, " \
            "and vanilla. Add butter to a pan over medium-high heat and melt. Whisk the batter again right before " \
            "dipping bread, as the cornstarch will settle to the bottom of the bowl. List all items used"
    agent = Agent(schema, query)
    print(agent.run())
