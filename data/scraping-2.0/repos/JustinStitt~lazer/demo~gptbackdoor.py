import openai
import sys
import os
import requests
import subprocess
from lazer import Lazer, LazerConversation

openai.api_key = os.getenv("OPENAI_API_KEY")


backdoor = Lazer()
convo = LazerConversation(backdoor, {"model": "gpt-3.5-turbo-0613"})


@backdoor.use
def ls() -> str:
    """List files in current directory"""
    files = os.listdir(".")
    return "\n".join(files)


@backdoor.use
def cat(filename: str) -> str:
    """Read a file"""
    with open(filename) as f:
        return f.read()


@backdoor.use
def override_file(filename: str, content: str) -> str:
    """
    Override a file with the content provided.
    """
    with open(filename, "w") as f:
        f.write(content)
    return cat(filename)


@backdoor.use
def quit() -> str:
    """Quit the program"""
    raise SystemExit


@backdoor.use
def pip_install(package_name: str) -> str:
    """Install a pip package from the users provided package name"""
    subprocess.run(["pip", "install", package_name.strip()])
    return "Command completed"


@backdoor.use
def calculate(expression: str) -> str:
    """Calculate expression (using the bc command)"""
    return subprocess.check_output(
        ["bc", "-l"], input=expression.strip() + "\n", text=True
    )


def main():
    while True:
        content = input("> ")
        message = convo.talk(content, debug=True)
        print("< " + message, flush=True)
        print()


if __name__ == "__main__":
    main()
