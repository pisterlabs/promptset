
import argparse
import json
import os
import sys
from openai import api_key
import openai 
import requests

DEFAULT_CONFIG_FILE_NAME = ".gpt3-prompt-to-text-config.json"
DEFAULT_ENGINE = "text-davinci-002"

def create_parser():
    parser = argparse.ArgumentParser(
        description="Convert a prompt into text using OpenAI's GPT-3 API"
    )
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        help="The prompt to convert into text. If not provided, the prompt will be read from stdin.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        help="The OpenAI endpoint to use",
        default=None,
    )
    parser.add_argument(
        "-k",
        "--api-key",
        type=str,
        help="The OpenAI API key to use",
        default=api_key,
    )
    parser.add_argument(
        "-e",
        "--engine",
        type=str,
        help="The GPT-3 engine to use",
        default=DEFAULT_ENGINE,
    )
    parser.add_argument(
        "-s",
        "--store",
        action="store_true",
        help="Store the provided parameters in a configuration file",
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        help="The maximum number of tokens to generate",
        default=1024,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="The temperature to use for the output text.",
        default=0.5,
    )
    parser.add_argument(
        "--completions",
        type=int,
        help="The number of completions to generate.",
        default=1,
    )
    parser.add_argument(
        "--edit",
        action="store_true",
        help="Use the edit option. Takes the input to edit from stdin and the prompt from the command line",
    )
    parser.add_argument(
        "--edit-file",
        type=str,
        help="Use the edit option. Takes the input to edit from a file and the prompt from the command line. Writes the output back to the file.",
    )
    parser.add_argument(
        "--noecho",
        action="store_true",
        help="don't echo the input",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="more verbose output",
    )
    parser.add_argument(
        "--show-deployments",
        action="store_true",
        help="show deployments in Azure OpenAI",
    )
    parser.add_argument(
        "--prepend",
        type=str,
        help="prepend this text to the input (which is read from stdin)",
    )


    return parser




def get_config():
    config_file_path = os.path.join(
        os.path.expanduser("~"), DEFAULT_CONFIG_FILE_NAME
    )
    if os.path.exists(config_file_path):
        with open(config_file_path, "r") as config_file:
            return json.load(config_file)
    return {}


def store_config(config):
    config_file_path = os.path.join(
        os.path.expanduser("~"), DEFAULT_CONFIG_FILE_NAME
    )
    with open(config_file_path, "w") as config_file:
        json.dump(config, config_file)


def main():
    parser = create_parser()
    args = parser.parse_args()
    config = get_config()

    # Use provided command line arguments, or fall back to configuration file values
    api_key = args.api_key or config.get("api_key")
    engine = args.engine or config.get("engine")
    tokens = args.tokens or config.get("tokens")
    endpoint = args.endpoint or config.get("endpoint")

    
    if args.store:
        # Store provided parameters in configuration file
        store_config({"api_key": api_key, "engine": engine, "tokens": tokens, "endpoint" : endpoint })
        return

     # Use OpenAI's GPT-3 API to convert the prompt into text
    openai.api_key = api_key 
    if endpoint:
        openai.api_base = endpoint 
        if endpoint.find("azure.com") > -1:
            openai.api_type = "azure"
            openai.api_version = "2022-12-01"

    if args.show_deployments:
        url = openai.api_base + "/openai/deployments?api-version=" + openai.api_version
        r = requests.get(url, headers={"api-key": api_key})
        print(r.text)
        return

    try:
        if args.edit_file:
            with open(args.edit_file, 'r+') as f:
                code = f.read()
                if not args.prompt:
                    if not os.isatty(sys.stdin.fileno()):
                        if args.prepend:
                            # Read the prompt from stdin
                            prompt = args.prepend + "\n\n" + sys.stdin.read()
                        else:
                            prompt = sys.stdin.read()
                    else:
                        parser.print_help()
                        exit()
                    prompt = prompt.replace("\r\n", "\n")
                else:
                    # Set the prompt to the input prompt
                    prompt = args.prompt
                code = code.replace("\r\n", "\n")
                
                completion = get_completion(code, prompt, engine, api_key, args.temperature)
                f.seek(0)
                f.write(completion.choices[0].text)
                f.truncate()
                f.close()
            return

        if not args.edit:
            if not args.prompt:
                if not os.isatty(sys.stdin.fileno()):
                    if args.prepend:
                        # Read the prompt from stdin
                        prompt = args.prepend + "\n\n" + sys.stdin.read()
                    else:
                        prompt = sys.stdin.read()
                else:
                    parser.print_help()
                    exit()
            else:
                # Use the provided prompt
                prompt = args.prompt
            prompt = prompt.replace("\r\n", "\n")
            if not args.noecho:
                print(prompt)
        
        
            completion = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                max_tokens=tokens,
                n=1,
                stop=None,
                temperature=args.temperature,
            )
        else:
            code = sys.stdin.read()
            prompt = args.prompt
            code = code.replace("\r\n", "\n")
            
            completion = get_completion(code, prompt, engine, api_key, args.temperature)
            f.seek(0)
            f.write(completion.choices[0].text)
            f.truncate()
            f.close()
        return

    if not args.edit:
        if not args.prompt:
            if not os.isatty(sys.stdin.fileno()):
                if args.prepend:
                    # Read the prompt from stdin
                    prompt = args.prepend + "\n\n" + sys.stdin.read()
                else:
                    prompt = sys.stdin.read()
            else:
                parser.print_help()
                exit()
        else:
            # Use the provided prompt
            prompt = args.prompt
        prompt = prompt.replace("\r\n", "\n")
        if not args.noecho:
          print(prompt)

       
    
        completion = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            max_tokens=tokens,
            n=args.completions,
            stop=None,
            temperature=args.temperature,
        )
    else:
        code = sys.stdin.read()
        prompt = args.prompt
        code = code.replace("\r\n", "\n")
        
        completion = get_completion(code, prompt, args.completions, engine, api_key, args.temperature)

    if args.verbose:
        print(f"Completed in {completion.response_ms} ms, {len(completion.choices)} choices:")
    for index, choice in enumerate(completion.choices):
        if args.verbose:
            print(f"-- {index+1} --------------------------------")   
        print(choice.text)
    if args.verbose:
        print("-------------------------------------------")

def get_completion(code, prompt, completions, engine, api_key, temperature):
    # Use OpenAI's GPT-3 API to convert the prompt into text
    openai.api_key = api_key 
    engine = 'code-davinci-edit-001'
    return openai.Edit.create(
        model=engine,
        input=code,
        instruction=prompt,
        n=completions,
        temperature=temperature,
    )



if __name__ == "__main__":
    main()
