import openai
import time
import os
import json
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
import black
import ast

openai.api_key = "INSERT_API_KEY"
openai.organization = "INSERT_ORG_NAME"

MAX_TOKENS = 15000
TOKENS_FOR_EACH_ITERATION = None
console = Console()
TOKENS_USED = 0

def welcome_message():
    console.print(Panel.fit("[bold magenta]Python Code Generator and Analyzer[/bold magenta]"))
    console.print("This program uses AI to either generate new Python code or analyze and improve existing code.")
    console.print("Please follow the prompts to proceed.\n")

def call_openai_api(messages, temperature):
    global TOKENS_USED
    try:
        console.log("Calling OpenAI API with the following messages:", json.dumps(messages, indent=4, ensure_ascii=False))
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            temperature=temperature,
            max_tokens=MAX_TOKENS - TOKENS_USED
        )
        if response.choices:
            console.log("OpenAI API call successful.")
            TOKENS_USED += response.usage["total_tokens"]
            code = response.choices[0].message['content']
            try:
                ast.parse(code)
                formatted_code = black.format_str(code, mode=black.FileMode())
                return formatted_code
            except SyntaxError:
                console.log("The received code could not be parsed as valid Python. Skipping formatting.")
                return code
            except Exception as e:
                console.log(f"An error occurred while formatting: {e}")
                return code
        else:
            console.log("OpenAI API call returned no choices.")
            return ""
    except Exception as e:
        console.log(f"An error occurred: {e}")
        return ""

def context_generator(create_new_program):
    initial_message = "You're an AI that generates Python code. Please provide an initial draft of the code in the form of a complete and runnable Python script." if create_new_program else "You're an AI that summarizes existing Python code. Provide both common, unique, and helpful ideas for implementation."
    return iter([
        initial_message,
        "You're an AI that reviews Python code for potential improvements and bugfixes. Provide detailed comments on any issues you find, along with recommended changes. Make sure to explain why these changes are necessary.",
        "You're an AI that is tasked to implement the recommended changes from the previous review step. Make sure the changes do not affect the intended functionality of the code.",
        "You're an AI that improves code readability and maintainability. Refactor the code to make it easier to understand and maintain, without changing its behavior or removing functionality.",
        "You're an AI that documents Python code according to PEP-8 style guidelines. Add extensive comments and docstrings to the code to ensure it is well-documented. Also add your autorship attribution in comments in the code.",
        "You're an AI that puts finishing touches on Python codebases and restores features removed by accident. Make sure everything looks good to go and then output the corrected codebase!"

    ])

def get_user_input():
    try:
        return input("Do you want to continue to the next step? (y/n): ")
    except Exception as e:
        console.log(f"An error occurred while getting user input: {e}")
        return None

def save_code(code, filename):
    try:
        with open(filename, 'w') as f:
            f.write(code)
        console.log(f"Code saved to {filename}")
    except Exception as e:
        console.log(f"An error occurred while saving the file: {e}")

def ensure_tokens_limit(messages, buffer_tokens=250):
    global TOKENS_USED
    while TOKENS_USED + buffer_tokens > MAX_TOKENS:
        removed_message = messages.pop(0)
        TOKENS_USED -= len(removed_message['content'])
    return messages

def get_initial_request(create_new_program):
    if create_new_program:
        try:
            return input("What code do you want me to write?: ")
        except Exception as e:
            console.log(f"An error occurred while getting user input: {e}")
            return None
    else:
        filename = input("Please enter the name of the file you want analyzed: ")
        try:
            with open(filename, 'r') as file:
                return file.read()
        except Exception as e:
            console.log(f"An error occurred while reading the file: {e}")
            return get_initial_request(create_new_program)

def get_operation_mode():
    try:
        return input("Create a new program or analyze an existing one? (new/existing): ").lower() == "new"
    except Exception as e:
        console.log(f"An error occurred while getting user input: {e}")
        return None

def process_iteration(context, messages):
    global initial_code
    console.log(f"\nIteration...")
    try:
        context_message = next(context, None)
        if context_message is not None:
            console.log(f"System directive: {context_message}")
            messages.append({"role": "system", "content": context_message})
        else:
            console.log("Reached the end of the cycle.")
            console.log("Final code:")
            console.print(Markdown(f"```python\n{initial_code}\n```"))
            return messages, initial_code, False
    except StopIteration:
        return messages, initial_code, False
    messages = ensure_tokens_limit(messages)
    improved_code = call_openai_api(messages, temperature=0.6)
    console.log("Assistant's response:")
    console.print(Markdown(f"```python\n{improved_code}\n```"))
    messages.append({"role": "assistant", "content": improved_code})
    initial_code = improved_code
    return messages, initial_code, True

def user_decision(improved_code):
    user_input = get_user_input()
    if user_input is None:
        return True
    user_input = user_input.lower()
    while user_input not in ['y', 'n', 's']:
        console.log("Invalid input. Please enter 'y' for yes, 'n' for no, or 's' to save the code.")
        user_input = get_user_input()
        if user_input is None:
            return True
        user_input = user_input.lower()
    if user_input == 'n':
        console.log("Stopping iterations. Final code:")
        console.print(Markdown(f"```python\n{improved_code}\n```"))
        return False
    elif user_input == 's':
        filename = input("Enter filename to save the code: ")
        save_code(improved_code, filename)
    time.sleep(2)
    return True

def execute_operation(create_new_program):
    global TOKENS_USED
    TOKENS_USED = 0
    context = context_generator(create_new_program)
    messages = [{"role": "system", "content": next(context)}]
    request = get_initial_request(create_new_program)
    if request is None:
        return
    console.log("Processing initial request... ")
    initial_code = call_openai_api(messages + [{"role": "user", "content": request}], temperature=0.5)
    console.log("Initial code:")
    console.print(Markdown(f"```python\n{initial_code}\n```"))
    messages.append({"role": "assistant", "content": initial_code})

    while True:
        messages, improved_code, should_continue = process_iteration(context, messages)
        if not should_continue:
            break
        if not user_decision(improved_code):
            break

def should_restart():
    user_input = input("Do you want to restart the process? (y/n): ").lower()
    while user_input not in ['y', 'n']:
        console.log("Please enter 'y' for yes or 'n' for no.")
        user_input = input("Do you want to restart? (y/n): ").lower()
    return user_input == 'y'

def main():
    welcome_message()
    while True:
        create_new_program = get_operation_mode()
        if create_new_program is None:
            continue
        execute_operation(create_new_program)
        if not should_restart():
            console.log("Goodbye!")
            break

if __name__ == "__main__":
    main()
