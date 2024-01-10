import openai
import instructor
from getpass import getpass
from openai import Client
client = Client(
    #api_key=getpass("Paste your openai api key: ")
)

import textwrap
import builtins

def wprint(*args, width=70, **kwargs):
    """
    Custom print function that wraps text to a specified width.

    Args:
    *args: Variable length argument list.
    width (int): The maximum width of wrapped lines.
    **kwargs: Arbitrary keyword arguments.
    """
    wrapper = textwrap.TextWrapper(width=width)

    # Process all arguments to make sure they are strings and wrap them
    wrapped_args = [wrapper.fill(str(arg)) for arg in args]

    # Call the built-in print function with the wrapped text
    builtins.print(*wrapped_args, **kwargs)

import time

def get_completion(message, agent, funcs, thread):
    """
    Executes a thread based on a provided message and retrieves the completion result.

    This function submits a message to a specified thread, triggering the execution of an array of functions
    defined within a func parameter. Each function in the array must implement a `run()` method that returns the outputs.

    Parameters:
    - message (str): The input message to be processed.
    - agent (OpenAI Assistant): The agent instance that will process the message.
    - funcs (list): A list of function objects, defined with the instructor library.
    - thread (Thread): The OpenAI Assistants API thread responsible for managing the execution flow.

    Returns:
    - str: The completion output as a string, obtained from the agent following the execution of input message and functions.
    """

    # create new message in the thread
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=message
    )

    # run this thread
    run = client.beta.threads.runs.create(
      thread_id=thread.id,
      assistant_id=agent.id,
    )

    while True:
      # wait until run completes
      while run.status in ['queued', 'in_progress']:
        run = client.beta.threads.runs.retrieve(
          thread_id=thread.id,
          run_id=run.id
        )
        time.sleep(1)

      # function execution
      if run.status == "requires_action":
        tool_calls = run.required_action.submit_tool_outputs.tool_calls
        tool_outputs = []
        for tool_call in tool_calls:
          wprint('\033[31m' + str(tool_call.function), '\033[0m')
          # find the tool to be executed
          func = next(iter([func for func in funcs if func.__name__ == tool_call.function.name]))

          try:
            # init tool
            func = func(**eval(tool_call.function.arguments))
            # get outputs from the tool
            output = func.run()
          except Exception as e:
            output = "Error: " + str(e)

          wprint(f"\033[33m{tool_call.function.name}: ", output, '\033[0m')
          tool_outputs.append({"tool_call_id": tool_call.id, "output": output})

        # submit tool outputs
        run = client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread.id,
            run_id=run.id,
            tool_outputs=tool_outputs
        )
      # error
      elif run.status == "failed":
        raise Exception("Run Failed. Error: ", run.last_error)
      # return assistant message
      else:
        messages = client.beta.threads.messages.list(
          thread_id=thread.id
        )
        message = messages.data[0].content[0].text.value
        return message

from typing import List
from pydantic import Field
from instructor import OpenAISchema

class ExecutePyFile(OpenAISchema):
    """Run existing python file from local disc."""
    file_name: str = Field(
        ..., description="The path to the .py file to be executed."
    )

    def run(self):
      """Executes a Python script at the given file path and captures its output and errors."""
      try:
          result = subprocess.run(
              ['python3', self.file_name],
              text=True,
              capture_output=True,
              check=True
          )
          return result.stdout
      except subprocess.CalledProcessError as e:
          return f"An error occurred: {e.stderr}"

class File(OpenAISchema):
    """
    Python file with an appropriate name, containing code that can be saved and executed locally at a later time. This environment has access to all standard Python packages and the internet.
    """
    chain_of_thought: str = Field(...,
        description="Think step by step to determine the correct actions that are needed to be taken in order to complete the task.")
    file_name: str = Field(
        ..., description="The name of the file including the extension"
    )
    body: str = Field(..., description="Correct contents of a file")

    def run(self):
        with open(self.file_name, "w") as f:
            f.write(self.body)

        return "File written to " + self.file_name

from openai import OpenAI
client = OpenAI(api_key=openai.api_key)

code_assistant_funcs = [File, ExecutePyFile]

code_assistant = client.beta.assistants.create(
  name='Code Assistant Agent',
  instructions="As a top-tier programming AI, you are adept at creating accurate Python scripts. You will properly name files and craft precise Python code with the appropriate imports to fulfill the user's request. Ensure to execute the necessary code before responding to the user.",
  model="gpt-4-1106-preview",
  tools=[{"type": "function", "function": File.openai_schema},
         {"type": "function", "function": ExecutePyFile.openai_schema},]
)


import subprocess
from enum import Enum
from pydantic import PrivateAttr
from typing import Literal

agents_and_threads = {
    "code_assistant": {
        "agent": code_assistant,
        "thread": None,
        "funcs": code_assistant_funcs
    }
}

class SendMessage(OpenAISchema):
    """Send messages to other specialized agents in this group chat."""
    recepient:Literal['code_assistant'] = Field(..., description="code_assistant is a world class programming AI capable of executing python code.")
    message: str = Field(...,
        description="Specify the task required for the recipient agent to complete. Focus instead on clarifying what the task entails, rather than providing detailed instructions.")

    def run(self):
      recepient = agents_and_threads[self.recepient]
      # if there is no thread between user proxy and this agent, create one
      if not recepient["thread"]:
        recepient["thread"] = client.beta.threads.create()

      message = get_completion(message=self.message, **recepient)

      return message

user_proxy_tools = [SendMessage]

user_proxy = client.beta.assistants.create(
  name='User Proxy Agent',
  instructions="""As a user proxy agent, your responsibility is to streamline the dialogue between the user and specialized agents within this group chat.
Your duty is to articulate user requests accurately to the relevant agents and maintain ongoing communication with them to guarantee the user's task is carried out to completion.
Please do not respond to the user until the task is complete, an error has been reported by the relevant agent, or you are certain of your response.""",
  model="gpt-4-1106-preview",
  tools=[
      {"type": "function", "function": SendMessage.openai_schema},
  ],
)


thread = client.beta.threads.create()
while True:
  user_message = input("User: ")

  message = get_completion(user_message, user_proxy, user_proxy_tools, thread)

  wprint(f"\033[34m{user_proxy.name}: ", message,'\033[0m')