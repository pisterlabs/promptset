#!/usr/bin/env python3

import os
import openai
import sys

import pyperclip

openai.api_key = os.getenv("OPENAI_API_KEY")


def send_request(req: str) -> str:
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
                "content": "Your work is to create a command for a generic Unix shell that fulfills a given request. You shall answer in one line, containing only the generated command that achieves the user's request. You may use pipes (|), redirections (>, <, >>, <<, >>>, etc.), logical operators (&&, ||, etc.), background execution (&) and other features that the shell provides if necessary. You may also use any generic commands that a Unix system may provide. If the user specifies to use a certain command(s), you must use them in your response."},
            {"role": "user", "content": f"Generate a command that does the following: {req}"}
        ]
    )

    return res['choices'][0]['message']['content']


def main():
    if len(sys.argv) == 1 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(f"Usage: {sys.argv[0]} [command]")
        print(f"Example: {sys.argv[0]} 'create a file named test.txt'")
        return

    command = " ".join(sys.argv[1:])

    result = send_request(command)
    print("Command: " + result)

    while True:
        confirmation = input(
            "Do you want to copy the command to the clipboard? [Y/n]: ")

        if confirmation.lower() == "y" or confirmation == "":
            pyperclip.copy(result)
            break
        elif confirmation.lower() == "n":
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


if __name__ == '__main__':
    main()
