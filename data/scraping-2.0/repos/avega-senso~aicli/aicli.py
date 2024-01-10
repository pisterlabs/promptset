#!/usr/bin/env python3
import os
from dotenv import load_dotenv
import openai
import sys
import argparse
import re

# Load .env file if it exists
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")

# Dynamically load patterns from the .env file
patterns = [os.getenv(f'PATTERN_{i}') for i in range(1, 100) if os.getenv(f'PATTERN_{i}')]

DEFAULT_MAX_TOKENS = os.getenv("MAX_TOKENS", None)
if DEFAULT_MAX_TOKENS:
    try:
        DEFAULT_MAX_TOKENS = int(DEFAULT_MAX_TOKENS)
    except ValueError:
        print("MAX_TOKENS in .env file is not a valid integer. Falling back to model default.")
        DEFAULT_MAX_TOKENS = None

SANITIZE_DEFAULT = os.getenv("SANITIZE", "false").lower() == "true"
DEFAULT_TEMPLATE = os.getenv("DEFAULT_TEMPLATE")

def sanitize_message(error_msg):
    sanitized_msg = error_msg
    for pattern in patterns:
        sanitized_msg = re.sub(pattern, '[REDACTED]', sanitized_msg)
    return sanitized_msg

def get_response_from_gpt3(prompt, system=None, model='gpt-3.5-turbo', temperature=0, max_tokens=None):
    if system:
        prompt = f"[SYSTEM: {system}] {prompt}"
    messages = [{'role': 'user', 'content': prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
        max_tokens=max_tokens
    )
    return response

def main(prompt=None, model='gpt-3.5-turbo', temperature=0, is_interactive=True, sanitize=False, max_tokens=None):
    original_prompt = prompt
    if sanitize:
        prompt = sanitize_message(prompt)
    
    try:
        response = get_response_from_gpt3(prompt, SYSTEM, model, temperature, max_tokens=max_tokens)
        for chunk in response:
            if 'choices' in chunk and len(chunk['choices']) > 0 and 'content' in chunk['choices'][0]['delta']:
                content = chunk['choices'][0]['delta']['content']
                print(content, end='', flush=True)
        print()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

def get_input_from_std_streams():
    input_data = sys.stdin.read().strip()
    if not input_data:
        # input_data = sys.stderr.read().strip()
        input_data = sys.stdin.read().strip()
    return input_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="ai", description="Interact with OpenAI.")
    parser.add_argument('prompt', type=str, nargs='?', default=None, help="Initial prompt to provide to OpenAI.")
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help="Model to use for the interaction.")
    parser.add_argument('--temperature', type=float, default=0, help="Temperature setting for the model's response.")
    parser.add_argument('--sanitize', action="store_true", default=SANITIZE_DEFAULT, help="Sanitize the message to redact sensitive information.")
    parser.add_argument('--template', type=str, default=None, help="Name of the predefined prompt template in .env to use (e.g., ERROR, COMMIT).")
    parser.add_argument('--system', type=str, default=None, help="Override the SYSTEM variable from the .env file.")
    parser.add_argument('--max_tokens', type=int, default=DEFAULT_MAX_TOKENS, help="Maximum number of tokens for the model's response.")
    
    args = parser.parse_args()

    # Get the SYSTEM variable from .env or from the argument
    SYSTEM = args.system if args.system else os.getenv("SYSTEM")
    if not SYSTEM:
        print("SYSTEM value not provided or found in .env")
        sys.exit(1)
    
    prompt_to_use = args.prompt

    if not prompt_to_use and (not sys.stdin.isatty() or not sys.stderr.isatty()):
        prompt_to_use = get_input_from_std_streams()
    elif not prompt_to_use:
        prompt_to_use = input("AI% ")

    template_name = args.template if args.template else DEFAULT_TEMPLATE
    if template_name:
        template_key = f'TEMPLATE_{template_name.upper()}'
        template_value = os.getenv(template_key)
        if not template_value:
            print(f"Template '{template_name}' not found in .env")
            sys.exit(1)
        if '{message}' in template_value:
            prompt_to_use = template_value.format(message=prompt_to_use)
        
    main(prompt=prompt_to_use, model=args.model, temperature=args.temperature, sanitize=args.sanitize, max_tokens=args.max_tokens)
