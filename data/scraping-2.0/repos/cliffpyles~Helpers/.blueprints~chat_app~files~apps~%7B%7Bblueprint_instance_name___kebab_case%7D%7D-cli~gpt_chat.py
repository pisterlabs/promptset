import openai
import sys
from pathlib import Path

def strip_cwd(filepath):
    cwd = Path.cwd()
    fullpath = Path(filepath).resolve()
    if fullpath.parts[:len(cwd.parts)] == cwd.parts:
        return str(fullpath.relative_to(cwd))
    else:
        return str(fullpath)

def read_file(file_path):
    """Read the content of a file."""
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def chat_with_gpt(config, responses):
    """
    Interacts with the GPT-3.5-turbo model based on the provided config and user responses.

    :param config: The loaded configuration dictionary containing prompts and other settings.
    :param responses: The user responses dictionary.
    :return: A tuple containing the GPT model's response and the command string to reproduce the context.
    """

    # Read the contents of files that have 'file' prompt type
    for prompt in config["prompts"]:
        prompt_key = prompt.get('key')
        prompt_type = prompt['type']
        response = responses.get(prompt_key)

        if prompt_type == 'file' and response is not None and response.strip() != '':
            filepath = responses[prompt_key]
            responses[f"{prompt_key}_filepath"] = strip_cwd(filepath)
            responses[f"{prompt_key}_content"] = read_file(filepath)
            del responses[prompt_key]

    # Initialize the messages list with the system message
    messages = [{"role": "system", "content": config["context"]}]

    # Add user responses as separate messages
    for k, v in responses.items():
        for prompt in config["prompts"]:
            prompt_key = prompt.get('key')
            prompt_type = prompt['type']
            response = responses[k]

            messages.append({"role": "user", "content": f"{k}: {v}"})

    # Predefined messages from the configuration
    for message in config["messages"]:
        messages.append({"role": "user", "content": message})

    # Send messages to ChatGPT and return the response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    chatgpt_response = response.choices[0].message['content'].strip()
    return chatgpt_response
