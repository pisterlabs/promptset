import argparse
import json
import openai
import platform
import os

parser = argparse.ArgumentParser(
    description="Looking for help? Simply ask this program to write your command for you!"
)

parser.add_argument(
    "helpme",
    nargs="*",
    help="The description of what you want to do.",
)

args = parser.parse_args()
if len(args.helpme) <= 0:
    print("Please provide a single explanation.")
    exit(1)

# convert the arguments into a single string
explanation = " ".join(args.helpme)
operating_system = platform.system()
if operating_system == "Darwin":
    operating_system = "MacOS"

model = "gpt-3.5-turbo"

if os.getenv("OPENAI_APIKEY") is not None:
    openai.api_key = os.getenv("OPENAI_APIKEY")
else:
    with open("secrets.json", "r") as f:
        secrets = json.load(f)["key"]
        openai.api_key = secrets

functions = [
    {
        "name": "output_command",
        "description": "Output the command to the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": f"The {operating_system} command to run.",
                },
                "explanation": {
                    "type": "string",
                    "description": f"The explanation of why the {operating_system} command you are providing is correct.",
                },
            },
        },
    }
]

function_call = {
    "name": "output_command",
}

messages = [
    {
        "role": "system",
        "content": f"""You are a function that outputs a {operating_system} command to the user doing exactly what they describe. The user will provide a description of what they want to do, and you will output the {operating_system} command to run. Also output an explanation of the command to the user.""",
    },
    {
        "role": "user",
        "content": f"""{explanation}""",
    },
]

completion = openai.ChatCompletion.create(
    model=model, messages=messages, functions=functions, function_call=function_call
)

message = completion.choices[0].message
arguments = message["function_call"]["arguments"]

try:
    command = json.loads(arguments)["command"]
    explanation = json.loads(arguments)["explanation"]
except:
    print("Sorry, I didn't understand that. Please try again.")
    print(f"Here's what the model output:\n{message}")
    exit(1)

print()
print()
print(f"{command}")
print()
print(f"{explanation}")
print()
