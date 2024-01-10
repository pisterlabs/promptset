import openai
import instructor
from getpass import getpass
from openai import Client
import textwrap
import builtins
import time
from typing import List
from pydantic import Field
from instructor import OpenAISchema
import subprocess
from enum import Enum
from pydantic import PrivateAttr
from typing import Literal

# Initialize the OpenAI Client
client = Client()

def wprint(*args, width=70, **kwargs):
    # Custom print function that wraps text for better readability in console
    wrapper = textwrap.TextWrapper(width=width)
    wrapped_args = [wrapper.fill(str(arg)) for arg in args]
    builtins.print(*wrapped_args, **kwargs)

def get_completion(message, agent, funcs, thread):
    # Function to handle the completion of tasks via the OpenAI thread
    # It manages the execution of tasks and retrieves the result from the agent
    message = client.beta.threads.messages.create(thread_id=thread.id, role="user", content=message)
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=agent.id)

    while True:
        # Loop to check the status of the run and process it accordingly
        while run.status in ['queued', 'in_progress']:
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            time.sleep(1)

        if run.status == "requires_action":
            # Handling required actions by executing relevant functions
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []
            for tool_call in tool_calls:
                wprint('\033[31m' + str(tool_call.function), '\033[0m')
                func = next(func for func in funcs if func.__name__ == tool_call.function.name)
                try:
                    func = func(**eval(tool_call.function.arguments))
                    output = func.run()
                except Exception as e:
                    output = "Error: " + str(e)

                wprint(f"\033[33m{tool_call.function.name}: ", output, '\033[0m')
                tool_outputs.append({"tool_call_id": tool_call.id, "output": output})

            run = client.beta.threads.runs.submit_tool_outputs(thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs)
        elif run.status == "failed":
            # Handling the case where the run fails
            raise Exception("Run Failed. Error: ", run.last_error)
        else:
            # Returning the final message from the assistant
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            message = messages.data[0].content[0].text.value
            return message

class ExecutePyFile(OpenAISchema):
    # Schema for executing a Python file
    file_name: str = Field(..., description="The path to the .py file to be executed.")

    def run(self):
        # Executes the specified Python file and captures the output
        try:
            result = subprocess.run(['python3', self.file_name], text=True, capture_output=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"An error occurred: {e.stderr}"

class File(OpenAISchema):
    # Schema for handling Python file operations
    chain_of_thought: str = Field(..., description="Think step by step to determine the correct actions.")
    file_name: str = Field(..., description="The name of the file including the extension")
    body: str = Field(..., description="Correct contents of a file")

    def run(self):
        # Writes the provided content to a file
        with open(self.file_name, "w") as f:
            f.write(self.body)
        return "File written to " + self.file_name

# Initialize OpenAI with the API key
from openai import OpenAI
client = OpenAI(api_key=openai.api_key)

# Define functions that will be used by the code assistant
code_assistant_funcs = [File, ExecutePyFile]

# Create a code assistant agent with specific instructions
code_assistant = client.beta.assistants.create(
    name='Code Assistant Agent',
    instructions="Create accurate Python scripts.",
    model="gpt-4-1106-preview",
    tools=[{"type": "function", "function": File.openai_schema},
           {"type": "function", "function": ExecutePyFile.openai_schema}]
)

class SendMessage(OpenAISchema):
    # Schema for sending messages to other agents
    recepient: Literal['code_assistant'] = Field(..., description="Recipient agent.")
    message: str = Field(..., description="Specify the task for the recipient agent.")

    def run(self):
        # Sends a message to the specified recipient agent and gets a response
        recipient = agents_and_threads[self.recepient]
        if not recipient["thread"]:
            recipient["thread"] = client.beta.threads.create()

        message = get_completion(message=self.message, **recipient)
        return message

# Define tools for the user proxy
user_proxy_tools = [SendMessage]

# Create a user proxy agent to facilitate communication between the user and the code assistant
user_proxy = client.beta.assistants.create(
    name='User Proxy Agent',
    instructions="Articulate user requests to relevant agents.",
    model="gpt-4-1106-preview",
    tools=[{"type": "function", "function": SendMessage.openai_schema}]
)

# Mapping of agents and threads for communication
agents_and_threads = {
    "code_assistant": {
        "agent": code_assistant,
        "thread": None,
        "funcs": code_assistant_funcs
    }
}

# Main loop to handle user input and get responses from the user proxy
thread = client.beta.threads.create()
while True:
    user_message = input("User: ")
    message = get_completion(user_message, user_proxy, user_proxy_tools, thread)
    wprint(f"\033[34m{user_proxy.name}: ", message, '\033[0m')
