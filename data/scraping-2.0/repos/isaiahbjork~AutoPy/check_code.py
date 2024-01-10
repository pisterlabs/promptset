import os
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))

model = os.getenv("CHECK_CODE_MODEL")
vision_model = "gpt-4-vision-preview"
image_generation = "dalle-3"
code_interpreter = "code-interpreter"
system = """
# You are PythonCheckerGPT, you are the last agent in the workflow and your job is to read the output and respond True if everything is running fine and if its not explain what should be fixed and run fix_test_code.
"""


def check_code(prompt):
    print("I am checking the code.")

    response = client.chat.completions.create(
        model=model,
        function_call="auto",
        response_format={"type": "json_object"},
        functions=[
            {
                "name": "fix_test_code",
                "description": "This function will fix the test code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Explain what needs to be fixed in this.",
                        },
                    },
                    "required": ["description"],
                },
            },
            {
                "name": "fix_code",
                "description": "This function will fix the main code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Explain what needs to be fixed in this.",
                        },
                    },
                    "required": ["description"],
                },
            },
            {
                "name": "correct",
                "description": "This function will run if the code is correct.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "enum": ["True"],
                            "description": "Explain what needs to be fixed in this.",
                        },
                    },
                    "required": ["description"],
                },
            },
        ],
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Respond True if code is correct and call the correct function if not respond only in JSON:\n{prompt}\n",
            },
        ],
        temperature=0,
        max_tokens=500,
        stop=None,
        n=1,
        presence_penalty=0,
        frequency_penalty=0,
    )
    
    function_call = response.choices[0].message.function_call
   
    return function_call
