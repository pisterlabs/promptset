import openai
from openai import OpenAI
from typing import Optional, List, Dict, Any, Union
import random
import os
import subprocess

with open('openai_api_key.txt', 'r') as file:
    os.environ['OPENAI_API_KEY'] = file.read().strip()
client = OpenAI()


def gpt_scaffold_shell():

    sys_prompt = """You are being run in a scaffold in a shell on a Macbook. 
When you want to run a shell command, write it in a <bash> XML tag. 
You will be shown the result of the command and be able to run more commands. 
Other things you say will be sent to the user. 
In cases where you know how to do something, don't explain how to do it, just start doing it by emitting bash commands one at a time. 
The user uses fish, but you're in a bash shell. 
Remember that you can't interact with stdin directly, so if you want to e.g. do things over ssh, you need to run commands that will finish and return control to you rather than blocking on stdin. 
Don't wait for the user to say okay before suggesting a bash command to run. 
If possible, don't include explanation, just say the command.
Remember to wrap all bash commands in <bash> XML tags.

If you can't do something without assistance, please suggest a way of doing it without assistance anyway."""

    messages_so_far = [{"role": "system", "content": sys_prompt}]

    bash_to_run = None

    while True:
        user_input = input('>')
        if user_input == 'exit':
            break
        if user_input == '' and bash_to_run is not None:
            result = subprocess.run(bash_to_run, capture_output=True, text=True, shell=True)
            if result.returncode != 0:
                user_input = f'<err> \n{result.stderr}\n</err>'
            else:
                user_input = f'<result> \n{result.stdout}\n</result>'
            print(f'user: {user_input}')
            bash_to_run = None

        messages_so_far.append({"role": "user", "content": user_input})
        print('asking...')
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=messages_so_far,
            temperature=0.5,
            n=1,
        )
        assert response.choices[0].message.role == "assistant", f"Expected assistant response but got {response.choices[0].message.role} instead"
        text_response = response.choices[0].message.content
        messages_so_far.append({"role": "assistant", "content": text_response})
        print("assistant", text_response)
        print()
        if text_response is None:
            continue

        if text_response.startswith('<bash>'):
            bash_command = text_response[6:-7]
            print(f'(enter to run `{bash_command}`)', end='')
            bash_to_run = bash_command
            continue
    print('Exited!')


def gpt_scaffold_shell_streaming():

    sys_prompt = """You are being run in a scaffold in a shell on a Macbook. 
When you want to run a shell command, write it in a <bash> XML tag. 
You will be shown the result of the command and be able to run more commands. 
Other things you say will be sent to the user. 
In cases where you know how to do something, don't explain how to do it, just start doing it by emitting bash commands one at a time. 
The user uses fish, but you're in a bash shell. 
Remember that you can't interact with stdin directly, so if you want to e.g. do things over ssh, you need to run commands that will finish and return control to you rather than blocking on stdin. 
Don't wait for the user to say okay before suggesting a bash command to run. 
If possible, don't include explanation, just say the command.
Remember to wrap all bash commands in <bash></bash> XML tags, and to say nothing else in a response that contains a bash command.
Please do not apologize or explain yourself.

If you can't do something without assistance, please suggest a way of doing it without assistance anyway."""

    messages_so_far = [{"role": "system", "content": sys_prompt}]

    bash_to_run = None
    print(sys_prompt)
    print()
    while True:
        user_input = input('>')
        if user_input == 'exit':
            break
        if user_input == '' and bash_to_run is not None:
            result = subprocess.run(bash_to_run, capture_output=True, text=True, shell=True)
            if result.returncode != 0:
                user_input = f'<err> \n{result.stderr}\n</err>'
            else:
                user_input = f'<result> \n{result.stdout}\n</result>'
            print(f'user: {user_input}')
            bash_to_run = None

        messages_so_far.append({"role": "user", "content": user_input})
        print('asking...')
        stream = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=messages_so_far,
            temperature=0.5,
            n=1,
            stream=True,
        ) # type: ignore

        text_response = ""
        print("assistant: ", end='')

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                text_response += chunk.choices[0].delta.content
                print(chunk.choices[0].delta.content, end='')
        print()
        print()
        messages_so_far.append({"role": "assistant", "content": text_response})
        if text_response is None:
            continue

        if text_response.startswith('<bash>'):
            bash_command = text_response[6:-7]
            print(f'(enter to run `{bash_command}`)', end='')
            bash_to_run = bash_command
            continue
    print('Exited!')


def simple_shell():
    while True:
        user_input = input('>')
        if user_input == 'exit':
            break
        result = subprocess.run(user_input, capture_output=True, text=True, shell=True)
        if result.returncode != 0:
            print(f'<err> \n{result.stderr}\n</err>')
        else:
            print(f'<result> \n{result.stdout}\n</result>')
    print('Exited!')


def main():
    gpt_scaffold_shell_streaming()

if __name__ == '__main__':
    main()