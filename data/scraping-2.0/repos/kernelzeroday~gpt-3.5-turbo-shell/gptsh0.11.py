import openai
import json
import subprocess
import logging
import logging.handlers
import sys
import os
import pty
from termcolor import colored
import argparse
import traceback
import tiktoken
import pandas as pd

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens


# Global messages list
messages = []


def process_command_output(output, max_tokens):
    output_tokens = num_tokens_from_string(output)

    if output_tokens <= max_tokens:
        return output
    else:
        return "The output is too large to display. Please try another command or adjust the output."



# Replace with your OpenAI API key
openai.api_key = "PUT_KEY_HERE"

def setup_logging(log_level):
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # File handler for logging
    file_handler = logging.FileHandler("bash_commands.log")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console handler for logging
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def chatgpt_query(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=150,
    )

    return response.choices[0].message['content'].strip()

def extract_json(input_string):
    stack = []
    result = []
    for i, c in enumerate(input_string):
        if c == '{':
            stack.append(i)
        elif c == '}' and stack:
            start = stack.pop()
            if not stack:  # This means we've closed a top-level JSON object
                result.append(input_string[start:i + 1])

    return result

def is_message_valid(message, max_tokens_per_message=512):
    if 'content' not in message:
        return True

    tokens = num_tokens_from_string(message['content'])
    return tokens <= max_tokens_per_message


def trim_messages(messages, max_tokens, goal_tokens, max_tokens_per_message=512):
    message_tokens = []

    # Filter out large messages
    messages = [message for message in messages if 'content' not in message or num_tokens_from_string(message['content']) <= max_tokens_per_message]

    for message in messages:
        if 'content' in message:
            tokens = num_tokens_from_string(message['content'])
            message_tokens.append((message, tokens))
        else:
            message_tokens.append((message, 0))

    total_tokens = sum(tokens for _, tokens in message_tokens)

    if total_tokens > max_tokens:
        message_tokens.sort(key=lambda x: x[1], reverse=True)
        diff = total_tokens - goal_tokens

        for idx, (message, tokens) in enumerate(message_tokens):
            if diff > tokens:
                diff -= tokens
                messages.pop(idx)
            else:
                messages.pop(idx)
                break

    return messages


def generate_bash_code(user_input, messages, context, prev_command):
    max_tokens = 4096
    goal_tokens = 3096

    # Trim the messages first
    try:
        messages = trim_messages(messages, max_tokens, goal_tokens)
    except Exception as e:
        tb = traceback.format_exc()
        logging.error("Error trimming messages: %s", e)
        logging.debug(tb)

        # Check if user_input is valid
    if not is_message_valid({"content": user_input}):
        logging.error("User input is too long. Please provide a shorter input.")
        return None


    # Construct the context using the trimmed messages
    message_context = [m['content'] for m in messages]
    context = ' '.join(message_context)

    # Calculate the number of tokens remaining for the user_input and context
    remaining_tokens = max_tokens - len(user_input.split()) - len(context.split())

    # Check if the remaining tokens are enough for the model to process
    if remaining_tokens < 10:
        raise ValueError("The remaining token space is too small to process the input.")

    # Call the OpenAI API with the updated context

    model_engine = "gpt-3.5-turbo"

    # Add the user input message
    messages.append({"role": "user", "content": f"{user_input}"})

    if context:
        context_message = f"You are a {args.os} shell assistant. Your role is to provide commands. The previous command '{prev_command}' had the following output:\n\n{context}\nPlease provide output in a machine-readable json object format with keys 'explanation', 'command', never use sudo or any commands that require interaction. do not add any extra notes. output only exactly as as instructed. if a command is not applicable give the reason why in the explanation field with an empty command value, do not repeat or clarify the previous command if you do not understand the current command. you must not use code blocks and only output raw valid json:"

        if is_message_valid({"content": context_message}):
            messages.insert(-1, {"role": "system", "content": context_message})
        else:
            logging.debug("Command output is too large. It will not be added to the context.")

    for _ in range(3):  # Set the maximum number of retries
        response = openai.ChatCompletion.create(
            model=model_engine,
            messages=messages,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.5,
        )
        response_text = response.choices[0].message['content'].strip()
        logging.debug(f"GPT-3.5-turbo response: {response_text}")
        try:

            json_content = extract_json(response_text)
            # Transform list to string
            json_content = ''.join(json_content)
            parsed_response = json.loads(json_content)

        except json.JSONDecodeError as e:
            try:
                # If the response is not valid JSON, try to transform it into an explanation json object
                parsed_response = {"explanation": response_text, "command": ""}
            except Exception as e:
                # If that fails, return an error
                tb = traceback.format_exc()
                logging.error("Failed to parse GPT-3.5-turbo response as JSON.")
                logging.error(f"Response: {response_text}")
                logging.error(f"Error: {e}")
                logging.error(f"Traceback: {tb}")
                parsed_response = {
                    "explanation": "I'm sorry, but I couldn't generate a valid response.",
                    "command": "",
                }

        if parsed_response["command"] != "":
            break  # If the command is not empty, exit the loop
        else:
            # If the command is empty, remind the AI of the format and try again
            messages.append({
                "role": "user",
                "content": "Please remember to provide output in a machine-readable json object format with keys 'explanation', 'command'. Try again."
            })


    return parsed_response

def execute_command(command):
    # if the command is not a string, return an error
    if not isinstance(command, str):
        return -1, "Command is not a string."
    if command == "":
        return -1, "Command is empty."
    try:
        # Append 'echo $?' to get the exit code of the command
        full_command = f"{command}; echo $?"
        # Start a new process with a pseudo-terminal
        pid, fd = pty.fork()
        if pid == 0:
            # Child process: execute the command in the new terminal
            os.execv("/bin/bash", ["/bin/bash", "-c", full_command])
        else:
            # Parent process: read the output of the child process
            output = []
            try:
                while True:
                    data = os.read(fd, 1024)
                    if not data:
                        break
                    # output.append(data.decode())
                    output.append(data.decode("utf-8", "ignore"))
            except OSError:
                pass

            output = "".join(output)
            # Extract the exit code from the last line of the output
            exit_code_line = output.rstrip().split("\n")[-1]
            try:
                exit_code = int(exit_code_line)
            except ValueError:
                exit_code = -1

            # Remove the exit code line from the output
            output = output[: output.rfind(exit_code_line)].rstrip()

            if exit_code != 0:
                error_output = f"An error occurred while executing the command:\n{output}"
                print(colored(error_output, 'red'))
                return error_output
            else:
                print(colored(output, 'white'))
                return output
    except Exception as e:
        tb = traceback.format_exc()
        error_output = f"An error occurred while executing the command:\n{e}\n{tb}"
        print(colored(error_output, 'red'))
        return error_output

def process_input(user_input, messages, context="", prev_command=""):
    if user_input.lower() == "exit":
        return

    logging.debug(f"User input: {user_input}")
    try:
        parsed_response = generate_bash_code(user_input, messages, context, prev_command)
        logging.debug(f"Parsed response: {parsed_response}")
    except Exception as e:
        tb = traceback.format_exc()
        logging.error("Failed to generate bash code.")
        logging.error(f"Error: {e}")
        logging.error(f"Traceback: {tb}")
        return

    try:
        bash_code = parsed_response["command"]
        explanation = parsed_response["explanation"]
    except KeyError as e:
        logging.error("Failed to extract command and explanation from GPT-3.5-turbo response.")
        return

    logging.debug(colored("Generated bash command for '{}':\n".format(user_input), 'green'))
    print(colored(explanation, 'yellow'))
    print(colored("\n{}".format(bash_code), 'cyan'))

    output = None
    if args.auto_exec or input("\nDo you want to execute this command? (Y/n): ").lower() != "n":
        output = execute_command(bash_code)

        if is_message_valid(output):
            messages.append({"role": "assistant", "content": output})
        else:
            output = "The generated output is too large to process. Please try again with a shorter command."
            print(colored("\n" + output, 'red'))

    return output, bash_code

def evaluate_output(output):
    messages = [
        {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI. Evaluate the output of the command for any errors."},
        {"role": "user", "content": f"Please evaluate this command output: {output}"}
    ]
    response = chatgpt_query(messages)

    return response.strip()

def process_file_input(input_file):
    with open(input_file, 'r') as file:
        commands = file.readlines()
    # initialize the context with the first command
    context = ""
    # for each command, process the input and update the context
    for command in commands:
        command = command.strip()
        print(colored(f"Processing command: {command}", 'blue'))
        # update the function call to get both output and command
        output, command = process_input(command, context)
        # only update the context if there's a command
        if output and command:
            context = output



def main(args):
    setup_logging(args.log_level)
    if args.input_file:
        process_file_input(args.input_file)
    else:
        global messages

        print(colored("Welcome to the GPT-3.5-turbo Shell!", 'green'))
        print("Type a command or type 'exit' to quit.")

        context = ""

        # Initialize the messages list with the starting system message
        messages.append({"role": "system", "content": f"You are a {args.os} shell assistant. Your role is to provide commands. Please provide output in a machine-readable json object format with keys 'explanation', 'command', never use sudo or any commands that require interaction. do not add any extra notes. output only exactly as as instructed. if a command is not applicable give the reason why in the explanation field with an empty command value, do not repeat or clarify the previous command if you do not understand the current command. you must not use code blocks and only output raw valid json:"})

        while True:
            try:
                user_input = input(colored("> ", 'cyan'))
                output, command = process_input(user_input, messages, context, prev_command=user_input)
                if output and command:
                    context = output
            except Exception as e:
                tb = traceback.format_exc()
                logging.error("An error occurred while processing the user input.")
                logging.error(e)
                logging.error(tb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-3.5-turbo Shell")
    parser.add_argument("-i", "--input-file", type=str, help="File with scripted commands")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument("--engine", type=str, default="gpt-3.5-turbo", help="OpenAI engine to use")
    # auto execute the command
    parser.add_argument("--auto_exec", action="store_true", help="Auto execute the command")
    # auto retry the command
    parser.add_argument("--retry", action="store_true", help="Auto retry the command")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries for the command")
    parser.add_argument("--max-tries", type=int, default=3, help="Max tries for the command")
    # tokens
    parser.add_argument("--tokens", type=int, default=100, help="Max tokens for the command")
    # color true or false
    parser.add_argument("--color", action="store_true", help="Color the output")
    # operating system
    parser.add_argument("--os", type=str, default="linux", help="Operating system")
    
    args = parser.parse_args()

    if args.input_file and not os.path.isfile(args.input_file):
        print("Input file not found. Exiting.")
        sys.exit(1)

    main(args)
