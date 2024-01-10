"""Hello Computer"""
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from role.developer import system, tools, available_functions

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
console = Console()


def main(messages: list = None, loop: bool = False):
    """Runs a conversation with the LLM from within the terminal."""
    if messages is None:
        messages = [system]
    if not loop:
        message = input("")
        print()
        if message.lower() == "bye" or message.lower() == "goodbye":
            return print("goodbye!\n")
        messages.append({"role": "user", "content": message})
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.0
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    messages.append(response_message)
    if tool_calls:
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(**function_args)
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response
                }
            )
            console.print(Markdown(f"`tool:`\n{function_name}"))
            print()
        return main(messages, loop=True)
    console.print(Markdown(f"`assistant:`\n{response_message.content}"))
    print()
    return main(messages)


main()
