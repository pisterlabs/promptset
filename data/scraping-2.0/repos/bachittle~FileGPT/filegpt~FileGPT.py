# this is the code for FileGPT.py
#!/usr/bin/env python

import sys
import os
import logging
import tempfile

from filegpt.openai_wrapper import get_openai_response
from filegpt.personalities import PERSONALITIES, get_personality, get_openai_personality

# from openai_wrapper import get_openai_response
# from personalities import PERSONALITIES, get_personality, get_openai_personality

import tiktoken
from typing import List

PERSONALITY = None

MODELS = [
    "gpt-3.5-turbo",
    "gpt-4"
]

TOKEN_LIMIT = {
    "gpt-3.5-turbo": 4096,
    "gpt-4": 8196 
}

def setup_logging():
    log_dir = tempfile.gettempdir()
    log_file = os.path.join(log_dir, "filegpt.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
    logging.info("Starting FileGPT")

def read_input(files: List[str] = None) -> str:
    if files:
        logging.info("Reading input from files: %s", files)
        contents = []
        for file in files:
            with open(file, "r") as f:
                contents.append(f.read())
        content = "\n".join(contents)
    else:
        logging.info("Reading input from stdin")
        if os.isatty(sys.stdin.fileno()):
            print("Current personality:", get_personality(PERSONALITY))
            print("Reading from stdin (press CTRL+D for linux/mac or Enter+CTRL+Z+Enter for windows to stop)...")
        content = sys.stdin.read()
    return content


def select_model(model_name:str) -> str:
    use_default = True
    for name in MODELS:
        if model_name == name:
            use_default = False
            break

    if use_default:
        logging.info("Using default model: %s", MODELS[0])
        model_name = MODELS[0]
    else:
        logging.info("Using specified model: %s", model_name)

    return model_name

def write_output(response):
    logging.info("Writing output")
    response_str = ""

    for chunk in response:
        chunk_msg = chunk['choices'][0]['delta']
        if 'content' in chunk_msg:
            sys.stdout.write(chunk_msg['content'])
            response_str += chunk_msg['content']
            sys.stdout.flush()
    
    return response_str

def process_text(model_name: str, input_files: List[str] = None) -> str:
    setup_logging()
    input_content = read_input(input_files)
    model_name = select_model(model_name)

    enc = tiktoken.encoding_for_model(model_name)

    logging.info("Input has %d tokens", len(enc.encode(input_content)))

    personality = get_openai_personality(PERSONALITY)
    messages = [{"role": "user", "content": input_content}]
    if personality is not None:
        messages.append(personality)
    response = get_openai_response(model_name, messages)
    resp_str = write_output(response)
    logging.info("FileGPT finished, response has %d tokens", len(enc.encode(resp_str)))
    
    return resp_str

def main():
    import argparse
    parser = argparse.ArgumentParser(description="FileGPT - A simple tool that autocompletes text files.")
    parser.add_argument("-f", "--file", help="Specify one or more text files as input.", type=str, nargs="+")
    parser.add_argument("-m", "--model", help="Specify the model to use for autocompletion.", type=str)
    parser.add_argument("-p", "--personality", help="Specify the personality to use for autocompletion.", type=str)

    parser.add_argument("personalities", help="Perform operations on available personalities.", type=str, nargs="?", choices=["p-ls"])
    args = parser.parse_args()

    if args.personalities:
        # handle personalities argument
        if args.personalities == "p-ls":
            # list available personalities
            print(f"Available personalities: {', '.join(PERSONALITIES)}")
            return
    
    global PERSONALITY
    PERSONALITY = get_personality(args.personality)
    process_text(args.model, args.file)


if __name__ == "__main__":
    main()

# (run FileGPT on itself and ask it the question below, using `filegpt -f FileGPT.py`)
# How can this code be improved?