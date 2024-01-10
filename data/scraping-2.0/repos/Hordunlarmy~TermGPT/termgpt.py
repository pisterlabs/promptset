#!/usr/bin/env python3
import openai
import os
import sys
import time
import readline
from pygments import highlight
from pygments.lexers import guess_lexer
from pygments.lexers import get_lexer_by_name
from pygments.formatters import TerminalFormatter
from pygments.token import (Keyword, Name, Comment, String, Error,
                            Number, Operator, Generic, Token, Whitespace)
from pygments import highlight
import colorama


# custom color scheme
COLOR_SCHEME = {
    Token:              ('gray',                 'gray'),
    Comment:            ('magenta',     'brightmagenta'),
    Comment.Preproc:    ('magenta',     'brightmagenta'),
    Keyword:            ('blue',                   '**'),
    Keyword.Type:       ('green',       '*brightgreen*'),
    Operator.Word:      ('**',                     '**'),
    Name.Builtin:       ('cyan',           'brightblue'),
    Name.Function:      ('blue',           'brightblue'),
    Name.Class:         ('_green_',        'brightblue'),
    Name.Decorator:     ('magenta',     'brightmagenta'),
    Name.Variable:      ('blue',           'brightblue'),
    String:             ('yellow',       'brightyellow'),
    Number:             ('blue',         'brightyellow')
}

# Init colors
colorama.init()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Check if the OpenAI API key is present
if not openai.api_key:
    print("OpenAI API key not found. Please set the"
          "OPENAI_API_KEY environment variable.")
    exit()


def colorize_code(code):
    """Syntax highlighting"""
    lexer = guess_lexer(code)
    formatter = TerminalFormatter(bg='dark', colorscheme=COLOR_SCHEME)
    return highlight(code, lexer, formatter)


def emulate_typing(message):
    """Print message letter by letter"""
    for ch in message:
        print(ch, end='', flush=True)
        time.sleep(0.002)
    print(flush=True)


def process_file(file_path):
    """Process content from a file"""
    with open(file_path, 'r') as file:
        return file.read()


def get_multiline_input():
    # Read the input until the Enter key is pressed
    user_input = ''
    while True:
        try:
            char = input()
            if char == 'quit' or char == 'exit':
                sys.exit()
            user_input += char
        except EOFError:
            break
    return user_input.strip()


messages = [{
    'role': 'system',
    'content': 'You are a helpful assistant.'
}]

# Main loop
file_processed = False

while True:
    try:
        if not file_processed and len(sys.argv) > 1:
            notes, file, *_ = sys.argv[1:]
            processed_file = process_file(file)
            question = f"{notes}\n{processed_file}"
            file_processed = True
        else:
            # Get user input
            sys.stdout.write(colorama.Fore.CYAN + 'You: ' +
                             colorama.Style.RESET_ALL)
            sys.stdout.flush()
            user_input = get_multiline_input()

            # Check if there is a file mentioned in the user input
            file_mentions = [
                word for word in user_input.split() if os.path.isfile(word)]
            if file_mentions:
                for file_mention in file_mentions:
                    with open(file_mention, 'r') as file:
                        file_content = file.read()
                        user_input = user_input.replace(
                            file_mention, file_content).strip()
                question = user_input
            else:
                question = user_input

        # Add user message to the conversation
        messages.append({
            'role': 'user',
            'content': question
        })

        # OpenAI API response
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        # Extract text from response
        answer = response.choices[0].message.content

        # Add ChatGPT message to the conversation
        messages.append({
            'role': 'assistant',
            'content': answer
        })

        # Print ChatGPT answer
        print(colorama.Fore.YELLOW + 'ChatGPT: ' +
              colorama.Style.RESET_ALL, end='')
        parse_answer = answer.split('```')
        if len(parse_answer) > 1:
            emulate_typing(parse_answer[0].strip()[:-1])
            for chunk in parse_answer[1:]:
                if chunk.strip():
                    code = '\n'.join(chunk.split('\n')[1:])
                    highlight_code = colorize_code(code)
                    if code.strip():
                        emulate_typing(highlight_code.strip())
                else:
                    emulate_typing('\n' + chunk.strip())
        else:
            emulate_typing(answer.strip())

    except KeyboardInterrupt:
        # Handle Ctrl+C or other interrupt signals
        print("\nExiting...")
        break

    # Handle other exceptions
    except Exception as e:
        print(colorama.Fore.YELLOW +
              f"ChatGPT: An error occurred - {e}" + colorama.Style.RESET_ALL)
