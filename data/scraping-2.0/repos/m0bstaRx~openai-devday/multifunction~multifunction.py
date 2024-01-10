import subprocess
import openai
import json

def write_file(filename, content):
    with open(filename, "w") as f:
        f.write(content)
    return "File written successfully"

def run_cmd(cmd):
    subprocess.run(cmd, shell=True)
    return "Command run successfully"

messages = [
    {
        "role": "user",
        "content": "Create a bouncing ball animation in raylib and build it and run it"
    }
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The name of the file",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write",
                    },
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_cmd",
            "description": "Run a command on the terminal",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "The command to run",
                    },
                },
                "required": ["cmd"],
            },
        },
    }
]

response = openai.chat.completions.create(
    #model="gpt-3.5-turbo-1106",
    model="gpt-4-1106-preview",
    messages=messages,
    tools=tools,
    tool_choice="auto",
    seed=142359931,
    temperature=0.5,
)

response_message = response.choices[0].message
tool_calls = response_message.tool_calls

messages.append(response_message)

if tool_calls:
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        function_response = globals()[function_name](**function_args)

        messages.append(
            {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            }
        )

for message in messages:
    if isinstance(message, dict) and message["content"] is None:
        message["content"] = ""
    if hasattr(message, "content") and message.content is None:
        message.content = ""

print(messages)

second_response = openai.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=messages,
)

print(second_response)