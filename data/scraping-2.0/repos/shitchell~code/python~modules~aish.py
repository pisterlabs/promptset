#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module complements a bash shell by using OpenAI's API to generate bash commands,
optionally executing them. Features:
- can run as an interactive shell
- can be used to generate a single bash command and optionally execute it

TODO:
- add support for fine-tuning the model using the user's bash history and any defined
  functions/aliases
"""

import openai as _openai
import shlex as _shlex
import time as _time
from typing import IO as _IO
from pathlib import Path as _Path
from textwrap import dedent as _dedent


def set_api_key(api_key: str) -> None:
    """Set the OpenAI API key.

    Args:
        api_key: The OpenAI API key.
    """
    _openai.api_key = api_key


def _test_api_key(raise_error: bool = False) -> bool:
    """
    Test the OpenAI API key by sending a request to the OpenAI API. If the request
    fails, an exception is raised.

    Args:
        error (bool, optional): Whether to raise an exception if the API key is invalid.
            Defaults to False.

    Returns:
        bool: Whether the API key is valid.
    """
    # Test the OpenAI API key by sending a request to the OpenAI API /completions
    # endpoint, thus using the API key, but not requiring any special permissions and
    # requiring the least amount of data/tokens.
    try:
        _openai.Completion.create(engine="davinci", prompt="test", max_tokens=1)
    except _openai.error.AuthenticationError:
        if raise_error:
            raise _openai.error.AuthenticationError("Invalid OpenAI API key")
        return False
    except _openai.error.InvalidRequestError:
        pass
    return True


def _generate_prompt(prompt: str) -> str:
    """
    Generate a prompt for the OpenAI Completion API suitable for generating bash
    commands.

    Args:
        prompt (str): The prompt to send to the OpenAI API.

    Returns:
        str: The generated prompt.
    """
    # Generate a prompt for the OpenAI Completion API suitable for generating bash
    # commands
    return _dedent(
        f"""
        # Prompt: Generate a bash command to find all files in the current directory
        # Response:
        # Search for all files in the current directory
        find . -type f
        # Prompt: Generate a bash command to show the current directory
        # Response:
        # Show the current directory
        ls
        # Prompt: {prompt}
    """
    )


def generate_command(
    prompt: str,
    temperature: float = 0.5,
    max_tokens: int = 100,
    engine="code-davinci-002",
    **kwargs,
) -> str:
    """Generate a bash command using the OpenAI API.

    Args:
        prompt (str): The prompt to send to the OpenAI API.
        temperature (float, optional): A value between 0-1 which determines how random
            the generated code is. Higher values result in more random code; lower
            values result in more predictable code. Defaults to 0.5.
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults
            to 100.
        **kwargs: Additional keyword arguments to pass to the OpenAI API.

    Returns:
        str: The generated bash command.
    """
    # Generate a bash command using the OpenAI API
    completion_args: dict[str, str | int | float] = {
        "engine": engine,
        "prompt": _generate_prompt(prompt),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": ["# Prompt:"],
    }
    completion_args.update(kwargs)
    completion = _openai.Completion.create(**completion_args)
    response: str = completion.choices[0].text
    # Strip any "# Response:" or "# Prompt:" lines from the response
    response = "\n".join(
        line
        for line in response.splitlines()
        if not (line.startswith("# Response:") or line.startswith("# Prompt:"))
    )
    return response


def _upload_file(filename: str, file: _IO | _Path | bytes | str) -> str:
    """Upload a file to the OpenAI API.

    Args:
        filename (str): The name of the file to upload.
        file (_IO | _Path | bytes | str): The file to upload.
            Bytes and strings are uploaded as-is.
            Paths are opened in binary mode and uploaded.
            File-like objects are read and uploaded.

    Returns:
        str: The ID of the uploaded file.
    """
    # Upload a file to the OpenAI API
    data: bytes = None
    if isinstance(file, _Path):
        with file.open("rb") as f:
            data = f.read()
    elif isinstance(file, str):
        data = file.encode()
    elif isinstance(file, bytes):
        data = file
    elif isinstance(file, _IO):
        data = file.read()
    return _openai.File.create(file=data, purpose="dataset", filename=filename).id


def __fine_tune(
    dataset: str, base_model: str = "davinci", engine_name: str = None, **kwargs
) -> str:
    """Fine-tune a model using the OpenAI API.

    Args:
        dataset (str): The dataset to use to fine-tune the model.
        base_model (str, optional): The base model to use. Defaults to "davinci".
        engine_name (str, optional): The name of the engine to create. If no engine
            name is provided, the computer's hostname and the currently logged in user
            will be used, e.g.: "aish_Johns-MacBook-Pro.local-john". Defaults to None.
        **kwargs: Additional keyword arguments to pass to the OpenAI API.

    Returns:
        str: The name of the engine that was created.
    """
    # Upload the dataset to the OpenAI API
    filename: str = f"{_time.strftime('%Y%m%d-%H%M%S')}.txt"
    dataset_id: str = _upload_file(filename, dataset)

    # Fine-tune a model using the OpenAI API
    if engine_name is None:
        import getpass as _getpass
        import socket as _socket

        engine_name = f"aish_{_socket.gethostname()}-{_getpass.getuser()}"
    engine_args: dict[str, str | int | float] = {
        "model": base_model,
        "dataset": dataset_id,
        "engine": engine_name,
    }
    engine_args.update(kwargs)
    _openai.Engine.create(**engine_args)
    return engine_name


def main():

    """Main entry point for the script."""
    import subprocess
    import os
    import readline
    import sys
    import time
    from argparse import ArgumentParser

    def prompt_to_execute(prompt: str = "Execute command? [y/n] ") -> bool:
        """Prompt the user to execute a command.

        Args:
            prompt (str, optional): The prompt to display. Defaults to
                "Execute command? [y/n] ".

        Returns:
            bool: True if the user entered "y" or "yes", False otherwise.
        """
        # Prompt the user to execute a command
        response: str = input(prompt).lower()
        if response in ["y", "yes"]:
            return True
        return False

    def print_command(command: str) -> None:
        """Print a bash command with comments greyed out.

        Args:
            command (str): The bash command to print.
        """
        for line in command.splitlines():
            line_parts = _shlex.split(line)
            fixed_line = ""
            added_color = False
            for part in line_parts:
                if part.startswith("#"):
                    fixed_line += f"\033[90m{part}"
                    added_color = True
                else:
                    fixed_line += f"{part} "
            # Remove any trailing whitespace
            fixed_line = fixed_line.rstrip()
            if added_color:
                fixed_line += "\033[0m"
            print(fixed_line)

    def run_command(command: str, delay: float = 0, confirm: bool = True) -> None:
        """Run a bash command.

        Args:
            command (str): The bash command to run.
        """
        if confirm and not prompt_to_execute():
            return
        if delay:
            time.sleep(delay)
        # Loop over and run any non-comment lines
        for line in command.splitlines():
            line_parts = _shlex.split(line)
            fixed_line = ""
            for part in line_parts:
                if part.startswith("#"):
                    break
                fixed_line += f"{part} "
            if fixed_line:
                # Remove any trailing whitespace
                fixed_line = fixed_line.rstrip()
                subprocess.run(fixed_line, shell=True)

    parser: ArgumentParser = ArgumentParser(
        description="Generate bash commands using the OpenAI API."
    )
    parser.add_argument(
        "-k",
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="The OpenAI API key to use.",
    )
    parser.add_argument(
        "prompt", help="The prompt to send to the OpenAI API.", nargs="?"
    )
    parser.add_argument(
        "-t",
        "--temperature",
        help="A value between 0-1 which determines how random the generated code is. "
        "Higher values result in more random code; lower values result in more "
        "predictable code. Defaults to 0.5.",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "-m",
        "--max-tokens",
        help="The maximum number of tokens to generate. Defaults to 100.",
        type=int,
        default=100,
    )
    parser.add_argument(
        "-e",
        "--engine",
        help="The engine to use. Defaults to 'code-davinci-002'.",
        default="code-davinci-002",
    )
    parser.add_argument(
        "-x",
        "--execute",
        default=False,
        help="Execute the generated command.",
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--delay",
        help="The number of seconds to wait before executing the command. Defaults "
        "to 0.",
        type=float,
        default=0,
    )
    parser.add_argument(
        "-y",
        "--yes",
        default=False,
        help="Automatically answer yes to any prompts.",
        action="store_true",
    )
    args = parser.parse_args()

    # Set the OpenAI API key
    if args.api_key is not None:
        _openai.api_key = args.api_key
    else:
        print("error: no API key provided or set by OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)

    # If "yes" was passed, then set "execute" to True and response to "y"
    if args.yes:
        args.execute = True
        response: str = "y"
    else:
        response: str = ""

    # If a prompt was given, then generate a command
    if args.prompt is not None:
        # Generate a bash command
        command: str = generate_command(
            args.prompt, args.temperature, args.max_tokens, args.engine
        )
        # Print the generated command
        print_command(command)

        # Execute the generated command if requested
        if args.execute:
            run_command(command, args.delay, not args.yes)
        elif prompt_to_execute():
            run_command(command, args.delay, not args.yes)
    else:
        # Enter interactive mode
        print("\033[1;32mAIshell\033[0m -- \033[32mType 'exit' to exit\033[0m")
        while True:
            # Get the user's prompt
            prompt: str = input("\033[2m>>\033[0m ")
            if prompt.lower() == "exit" or prompt.lower() == "quit":
                break

            # Generate a bash command
            command: str = generate_command(
                prompt, args.temperature, args.max_tokens, args.engine
            )
            # Print the generated command
            print_command(command)

            # Execute the generated command if requested
            if args.execute:
                run_command(command, args.delay, not args.yes)
            elif prompt_to_execute():
                run_command(command, args.delay, not args.yes)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print()
