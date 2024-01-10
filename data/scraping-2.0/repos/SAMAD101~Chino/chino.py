#!/usr/bin/python
import typer
import os
import openai
import json


openai.api_key = os.getenv("OPEN_API_KEY")

def get_response(prompt):
    """Get a response from GPT-3"""
    prompt_info = {
        "prompt": prompt,
    }
    return json.dumps(prompt_info)

def run_conversation(prompt) -> str:
    # Step 1: send the conversation and available functions to GPT
    messages = [{"role": "user", "text": prompt}]
    functions = [
        {
            "name": "get_response",
            "description": "Get a response from GPT-3",
            "parameters": {
                "type": "string",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Prompt for generating response by Chino using OpenAI's GPT-3 API",
                    },
                },
            },
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    response_message = response["choices"][0]["message"]["content"]

    # Step 2: check if GPT wanted to call a function
    if response_message.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_response": get_response(),
        }  # only one function in this example, but you can have multiple
        function_name = response_message["function_call"]["name"]
        function_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        function_response = function_to_call(
            location=function_args.get("response"),
        )

        # Step 4: send the info on the function call and function response to GPT
        messages.append(response_message)  # extend conversation with assistant's reply
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
        )  # get a new response from GPT where it can see the function response
        return second_response


def main(prompt: str = typer.Option(None, '-p', '--prompt', help="Prompt for ChatGPT")):
    if prompt is not None:
        response = run_conversation(prompt)
        typer.echo(response)
    else:
        typer.echo("Chino is Happy!")

if __name__ == "__main__":
    typer.run(main)
