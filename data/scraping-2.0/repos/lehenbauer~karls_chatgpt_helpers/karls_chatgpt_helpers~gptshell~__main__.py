#!/usr/bin/env python3

import argparse
import code
import json
import os
import readline
import requests
import subprocess
import tempfile
import time

import openai

import karls_chatgpt_helpers

PROMPT = "gpt> "

def run_editor():
    editor = os.environ.get('EDITOR', 'vi')
    with tempfile.NamedTemporaryFile(mode='w+', delete=True) as tmpfile:
        tmpfile.close()  # close the file so that the editor can open it
        # open the file in the user's preferred editor
        os.system(f'{editor} {tmpfile.name}')
        if os.path.exists(tmpfile.name):
            # read the contents of the temporary file
            with open(tmpfile.name, 'r') as f:
                contents = f.read()
            # delete the temporary file
            os.unlink(tmpfile.name)
            return contents.strip()
        else:
            return ''

def show_history(history):
    for row in history:
        print(row)

def converse(g):
    # Prompt the user for input in a loop
    while True:
        try:
            role = 'user'
            print()
            line = input(PROMPT)
            print()

            if line.startswith('%'):
                if line.startswith('%list'):
                    print(g.history)
                    continue
                elif line.startswith('%edit') or line.startswith('%sysedit'):
                    role = 'system' if line.startswith('%sysedit') else 'user'
                    # call the run_editor function to edit the input
                    line = run_editor()
                    # no continue here, we want to fall through and send the
                    # user's input to the chat API
                elif line.startswith('%load'):
                    # Code for %load command
                    filename = line.split()[1]
                    g.load(filename)
                    continue
                elif line.startswith('%save'):
                    # Code for %save command
                    filename = line.split()[1]
                    g.save(filename)
                    continue
                elif line.startswith('%yload'):
                    # Code for %load command
                    filename = line.split()[1]
                    g.load_yaml(filename)
                    continue
                elif line.startswith('%ysave'):
                    # Code for %ysave command
                    filename = line.split()[1]
                    g.save_yaml(filename)
                    continue
                elif line.startswith('%jload'):
                    # Code for %jload command
                    filename = line.split()[1]
                    g.load_json(filename)
                    continue
                elif line.startswith('%jsave'):
                    # Code for %jsave command
                    filename = line.split()[1]
                    g.save_json(filename)
                    continue
                elif line.startswith('%history'):
                    # Code for %history command
                    show_history(g.history)
                    continue
                elif line.startswith('%!'):
                    # Code for shell escape
                    command = line[2:].strip()
                    subprocess.run(command, shell=True)
                    continue
                elif line.startswith('%interact'):
                    # Code for interactive Python interpreter
                    print("Entering Python interpreter interactively... Send EOF to resume, exit() or quit() to exit.")
                    code.interact(local=locals())
                    continue
                elif line.startswith('%exit'):
                    # Code for exit command
                    break

                else:
                    print("Unrecognized % command, % commands are %list, %edit, %sysedit, %load, %save, %jload, %jsave, %yload, %ysave, %!, %interact and %exit")
                    continue

            if line.startswith('s:'):
                role = 'system'
                line = line[2:]

            if line == '':
                continue

            # Send the user input to ChatGPT
            g.streaming_chat(line, role=role)
        except EOFError:
            break
        except KeyboardInterrupt:
            print('interrupt')
            continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--system-file', help='the system prompt file to use')
    parser.add_argument('-i', '--load', help='load a session from a file')
    parser.add_argument('-w', '--save', help='save the session to a session file')
    parser.add_argument('-m', '--model', type=str, default="gpt-3.5-turbo", help='Model used for response generation')
    parser.add_argument('-t', '--temperature', type=float, default=0.7, help='Temperature for response generation')

    args = parser.parse_args()

    karls_chatgpt_helpers.openai_api_key_set_or_die()

    g = karls_chatgpt_helpers.GPTChatSession(
        model=args.model,
        temperature=args.temperature,
        debug=False
    )
    if args.system_file:
        with open(args.system_file, 'r') as f:
            system_prompt = f.read()
        g.streaming_chat(system_prompt, role='system')
    if args.load:
        g.load(args.load)
    converse(g)
    if args.save:
        g.save(args.save)

