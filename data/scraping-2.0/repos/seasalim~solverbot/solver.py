import openai
import json
import requests
import os
import logging
import time
from rich import print
import argparse

#
# Command execution API
#


class Command:
    headers = {'Content-Type': 'application/json'}
    server_url = 'http://localhost:3000/command'

    @staticmethod
    def request(command: str, parameter: str, commentary: str):
        data = {"command": command, "parameter": parameter,
                "commentary": commentary}
        logging.debug(f"Sending command request: {json.dumps(data)}")

        response = requests.post(
            Command.server_url, data=json.dumps(data), headers=Command.headers)

        logging.debug(
            f"Command response: {response.status_code}: {response.json()}")

        return json.dumps(response.json())

    @staticmethod
    def generate(args: str):
        size = args.get("size")
        return Command.request('generate', size, "Generating maze")

    @staticmethod
    def move(args: str):
        direction = args.get("direction")
        commentary = args.get("commentary")
        return Command.request('move', direction, commentary)

#
# Conversation execution
#


def run_conversation(autorun):
    #
    # CONFIGURATION
    #
    gpt_model = "gpt-3.5-turbo-0613"
    maze_size = 7  # Starting size of maze
    wait_time = 2  # Adjust the wait time between requests
    logging.basicConfig(level=logging.ERROR)  # Set the log level

    openai.api_key = os.getenv("OPENAI_API_KEY")

    messages = [{"role": "user", "content": f"Your task is to find your way out of a maze. You can only move in directions where there is a passageway. You can only use the available functions to interact with the user. First, generate a size {maze_size} maze, then move until you find the exit. Move with purpose, don't wander aimlessly. Give some interesting, amusing or funny commentary as you explore. Stop when you locate an exit."}]

    functions = [
        {
            "name": "maze_generate",
            "description": "Generate a maze of specified size",
            "parameters": {
                "type": "object",
                "properties": {
                    "size": {
                        "type": "string",
                        "description": "The size of the maze",
                    },
                },
                "required": ["size"],
            },
        },
        {
            "name": "maze_move",
            "description": "Move in the specified direction",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "description": "Can be north, south, east or west",
                    },
                    "commentary": {
                        "type": "string",
                        "description": "Some interesting, witty, or funny commentary about the location and exploration so far"
                    }
                },
                "required": ["direction"],
            },
        },
    ]

    available_functions = {
        "maze_generate": Command.generate, "maze_move": Command.move}

    while True:
        logging.debug(f"Sending message: {messages}")

        # Send message to GPT
        response = openai.ChatCompletion.create(
            model=gpt_model, messages=messages, functions=functions, function_call="auto")

        response_message = response["choices"][0]["message"]

        logging.debug(f"Received response: {response_message}")

        # Check if GPT wanted to call a function
        if response_message.get("function_call"):
            function_name = response_message["function_call"]["name"]
            function_to_call = available_functions[function_name]
            function_args = json.loads(
                response_message["function_call"]["arguments"])

            print(f"[cyan]AGENT:[/cyan] {function_name} {function_args}")

            if autorun:
                time.sleep(wait_time)
            else:
                input("Hit ENTER to confirm agent command")

            function_response = function_to_call(function_args)
            print(f"[red]RESPONSE:[/red] {function_response}\n")

            # Extend the conversation with GPT reply
            messages.append(response_message)

            # Extend the conversation with the function response
            messages.append(
                {"role": "function", "name": function_name, "content": function_response})
        else:
            print(f"[cyan]ASSISTANT:[/cyan]: {response_message}")
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Check for autorun command line parameter.")

    parser.add_argument('--autorun',
                        action='store_true',
                        help='The autorun command line parameter')

    args = parser.parse_args()
    print(f"Autorun: {args.autorun}\n")

    run_conversation(args.autorun)
