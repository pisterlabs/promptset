import os
import subprocess
from logging import Logger  # Just so the linter doesn't complain
import threading
import itertools
import time
import sys
import requests

import openai
from jinja2 import Environment, PackageLoader

from .config import get_api_key, get_model, set_api_key, set_model


class GitRepositoryError(Exception):
    """Custom exception for Git repository errors."""

    pass


class OpenAIKeyError(Exception):
    """Custom exception for OpenAI API key errors."""


class Llama2ServerError(Exception):
    """Custom exception for Llama2 server errors."""

    pass


def is_git_repository(directory: str) -> bool:
    """
    Check if a directory is a Git repository.

    Parameters
    ----------
    directory : str
        The path to check.

    Returns
    -------
    bool
        True if the directory is a Git repository, False otherwise.
    """
    print(directory)
    return os.path.isdir(os.path.join(directory, ".git"))


def run_git_diff(directory: str) -> str:
    """
    Run the `git diff --cached` command and return the result.

    Parameters
    ----------
    directory : str
        The directory to run the command in.

    Returns
    -------
    str
        The output of the command.
    """
    # TODO: Write unit tests for this function
    return subprocess.run(
        ["git", "diff", "--cached"], cwd=directory, capture_output=True, text=True
    ).stdout


def run_git_log(directory: str) -> subprocess.CompletedProcess:
    """
    Run 'git log' command in a specified directory.

    Parameters
    ----------
    directory : str
        The directory to run the command in.

    Returns
    -------
    subprocess.CompletedProcess
        The result of the subprocess run command.
    """
    return subprocess.run(
        ["git", "log", "--pretty=%B", "-n", "10"],
        cwd=directory,
        capture_output=True,
        text=True,
    )


def check_git_status(directory: str) -> bool:
    """
    Check if there are changes staged for commit in the Git repository.

    Parameters
    ----------
    directory : str
        The path to the Git repository.

    Returns
    -------
    bool
        True if there are changes staged for commit, False otherwise.
    """
    result = subprocess.run(
        ["git", "status"], cwd=directory, capture_output=True, text=True
    )
    return "no changes added to commit" not in result.stdout


def get_commit_messages(directory: str) -> str:
    """
    Fetch the last 10 commit messages from the Git repository.

    Parameters
    ----------
    directory : str
        The path to the Git repository.

    Returns
    -------
    str
        The commit messages.

    Raises
    ------
    GitRepositoryError
        If the specified directory is not a Git repository.
    """
    if not is_git_repository(directory):
        raise GitRepositoryError("The specified directory is not a Git repository.")

    result = run_git_log(directory)

    if result.returncode != 0:
        raise GitRepositoryError("Unable to get commit messages.")

    return result.stdout


def generate_commit_message(
    diff: str, start: str, ticket: str, api_key: str, model: str, llama2_url: str = None
) -> str:
    """
    Generate a commit message.

    Parameters
    ----------
    diff : str
        The git diff to use.
    start : str
        The start of the commit message.
    ticket : str
        The GitHub issue to link in the commit message.
    api_key : str
        The OpenAI API key.
    model : str
        The OpenAI GPT model to use.
    llama2_url: str
        (Optional) The URL of the Llama2 server to use.

    Returns
    -------
    str
        The generated commit message.

    Raises
    ------
    OpenAIKeyError
        If the OpenAI API key is not set.
    """
    # Perform guard check for the OpenAI API key
    if not api_key:
        raise OpenAIKeyError("OpenAI API key not set.")

    # Generate the prompt
    system_msg = render_template("", "system")
    user_msg = render_template(diff, "user")

    # Generate the commit message
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # Determine which service to use
    if llama2_url:
        # Call the Llama2 server
        response = call_llama2_server(llama2_url, messages)
        print(response)
        response = response["choices"][0]["message"]["content"].strip()
    else:
        # Use OpenAI's service
        openai.api_key = api_key
        response = openai.ChatCompletion.create(model=model, messages=messages)
        response = response.choices[0].message.content.strip()

    # Split the response by newline and store the result
    split_response = response.split("\n", 1)

    # Check if split_response contains at least 2 elements
    if len(split_response) > 1:
        response = split_response[1] if start else split_response[0]
    else:
        # If there is no newline, the whole response is either the start or the generated message
        if not start:
            start = response
        # If start is already set, we leave response as is, or set it to an empty string
        else:
            response = ""

    # Render and return the template
    return render_final_template(start, response, ticket).strip()


def call_llama2_server(url: str, messages: list) -> dict:
    """
    Call the Llama2 server.

    Parameters
    ----------
    url : str
        The URL of the Llama2 server.
    messages : list
        The messages to send to the server.

    Returns
    -------
    dict
        The response from the server.

    Raises
    ------
    Llama2ServerError
        If an error occurs while connecting to the server.
    """
    try:
        response = requests.post(url, json={"messages": messages})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Llama2ServerError(
            f"An error occurred while connecting to the Llama2 server: {e}"
        )


def render_template(message: str, template_name: str) -> str:
    """
    Render the commit message template.

    Parameters
    ----------
    message : str
        The commit message to render.
    template_name : str
        The name of the template to use.

    Returns
    -------
    str
        The rendered commit message.
    """
    # TODO: Write unit tests for this function

    # Render any escape sequences in the message
    message.encode().decode("unicode_escape")

    # Render and return the template
    jinja = Environment(loader=PackageLoader("magic_commit", "templates")).get_template(
        f"{template_name}.jinja"
    )
    return jinja.render(diff=message)


def render_final_template(start: str, generated_message: str, ticket: str) -> str:
    """
    Render the final commit message template.

    Parameters
    ----------
    start : str
        The start of the commit message.
    generated_message : str
        The generated commit message.
    ticket : str
        The GitHub issue to link in the commit message.

    Returns
    -------
    str
        The rendered commit message.
    """
    # Render any escape sequences in the message
    generated_message.encode().decode("unicode_escape")

    # Render and return the template
    jinja = Environment(loader=PackageLoader("magic_commit", "templates")).get_template(
        "final.jinja"
    )
    return jinja.render(start=start, generated_message=generated_message, ticket=ticket)


def animate_loading(stop_event: threading.Event) -> None:
    """
    Display a simple loading animation with dots until a stop_event is set.

    Parameters
    ----------
    stop_event : threading.Event
        An event to signal when the animation should stop.
    """
    for c in itertools.cycle([".  ", ".. ", "..."]):
        if stop_event.is_set():
            sys.stdout.write("\r       \r")  # Clear the line
            sys.stdout.flush()
            break
        sys.stdout.write("\rLoading" + c)
        sys.stdout.flush()
        time.sleep(0.5)


def run_magic_commit(
    directory: str,
    start: str,
    ticket: str,
    api_key: str,
    model: str,
    show_loading_message: bool,
    llama2_url: str = None,
) -> str:
    """
    Generate a commit message and return it.

    Parameters
    ----------
    directory : str
        The path to the Git repository.
    start : str
        The start of the commit message.
    ticket : str
        The GitHub issue to link in the commit message.
    api_key : str
        The OpenAI API key.
    model : str
        The OpenAI GPT model to use.
    show_loading_message : bool
        Whether or not to show the loading animation.
    llama2_url: str
        (Optional) The URL of the Llama2 server to use.

    Returns
    -------
    str
        The generated commit message.
    """
    # Create a threading event to signal when to stop the loading animation
    if show_loading_message:
        stop_loading = threading.Event()
        loading_thread = threading.Thread(target=animate_loading, args=(stop_loading,))
        loading_thread.start()

    try:
        diff = run_git_diff(directory)
        if not check_git_status(directory):  # Check if there are staged changes
            return "⛔ Warning: No staged changes detected. Please stage some changes before running magic-commit."
        commit_message = generate_commit_message(
            diff, start, ticket, api_key, model, llama2_url
        )
    finally:
        # Ensure the loading animation stops
        if show_loading_message:
            stop_loading.set()
            loading_thread.join()

    return commit_message
