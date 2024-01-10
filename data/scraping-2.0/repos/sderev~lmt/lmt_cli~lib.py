import json
import os
import sys
from pathlib import Path

import click
import openai
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.theme import Theme

from lmt_cli import gpt_integration as openai_utils

from .templates import handle_template

BLUE = "\x1b[34m"
RED = "\x1b[91m"
RESET = "\x1b[0m"


def prepare_and_generate_response(
    system: str,
    template: str,
    model: str,
    emoji: bool,
    prompt_input: str,
    temperature: float,
    tokens: bool,
    no_stream: bool,
    raw: bool,
    debug: bool,
):
    """
    Handles the parameters.
    """
    if not system:
        system = ""

    if template:
        system, prompt_input, model_template = handle_template(
            template, system, prompt_input, model
        )

    if not model:
        model = model_template

    if emoji:
        system = add_emoji(system)

    prompt = openai_utils.format_prompt(system, prompt_input)

    if debug:
        display_debug_information(prompt, model, temperature)

    if tokens:
        display_tokens_count_and_cost(prompt, model)

    stream = not no_stream

    content, response_time, response = generate_response(
        model,
        prompt,
        raw,
        stream,
        temperature,
    )

    return content, response_time, response


def add_emoji(system: str) -> str:
    """
    Adds an emoji to the system message.
    """
    emoji_message = (
        "Add plenty of emojis as a colorful way to convey emotions. However, don't"
        " mention it."
    )
    system = system.rstrip()

    if system == "":
        return emoji_message

    if not system.endswith("."):
        system += "."
    return system + " " + emoji_message


def get_config_path() -> Path:
    """
    Gets the path to the config file.
    """
    config_path = Path.home() / ".config/lmt/config.json"
    return config_path


def load_config() -> dict:
    """
    Loads the config file.
    """
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.touch(exist_ok=True)

    try:
        with open(config_path, "r", encoding="UTF-8") as file:
            config = json.load(file)
    except json.decoder.JSONDecodeError:
        config = {}

    return config


def save_config(config: dict) -> None:
    """
    Saves the config file.
    """
    config_path = get_config_path()
    with open(config_path, "w", encoding="UTF-8") as config_path:
        json.dump(config, config_path, indent=4)


def get_markdown_code_block_theme() -> str:
    """
    Gets the markdown code block theme from the config file.
    """
    config = load_config()
    config.setdefault("code_block_theme", "monokai")
    save_config(config)

    return config["code_block_theme"]


def get_markdown_inline_code_theme() -> str:
    """
    Gets the markdown inline code theme from the config file.
    """
    config = load_config()
    config.setdefault("inline_code_theme", "blue on black")
    save_config(config)

    return config["inline_code_theme"]


def generate_response(
    model: str = "gpt-3.5-turbo",
    prompt: str = None,
    raw: bool = False,
    stream: bool = True,
    temperature: float = 1,
):
    """
    Generates a response from a ChatGPT.
    """
    openai.api_key = get_api_key()

    if not openai.api_key:
        click.echo(
            f"{click.style('Error:', fg='red')} You need to set your OpenAI API key."
        )
        click.echo("You can do so by running:", nl=False)
        click.echo(f"  {click.style('lmt key set', fg='blue')}\n")
        sys.exit(1)

    # Theming for Rich Markdown
    code_block_theme = get_markdown_code_block_theme()
    inline_code_theme = get_markdown_inline_code_theme()
    custom_theme = Theme({"markdown.code": inline_code_theme})

    console = Console(theme=custom_theme)
    markdown_stream = ""
    with Live(markdown_stream, console=console, refresh_per_second=25) as live:
        # Allows rich markdown formatted stream in real time
        def update_markdown_stream(chunk: str) -> None:
            nonlocal markdown_stream
            markdown_stream += chunk
            if raw:
                print(chunk, end="")
            else:
                rich_markdown_stream = Markdown(
                    markdown_stream,
                    code_theme=code_block_theme,
                )
                live.update(rich_markdown_stream)

        try:
            content, response_time, response = openai_utils.chatgpt_request(
                api_key=openai.api_key,
                prompt=prompt,
                model=model,
                stream=stream,
                temperature=temperature,
                update_markdown_stream=update_markdown_stream,
            )

            # This is temporary to ensure that the last line always ends with a newline
            # This will be removed when refactored
            if not content.endswith("\n"):
                content += "\n"
            #############################

            if not stream:
                print(content)

        except openai.error.RateLimitError as error:
            click.echo(f"{RED}Error:{RESET} {error}", err=True)
            openai_utils.handle_rate_limit_error()
            sys.exit(1)

        except openai.error.AuthenticationError:
            openai_utils.handle_authentication_error()
            sys.stderr.write("\nYou can set your API key by running: ")
            sys.stderr.write(f"{BLUE}lmt key set{RESET}\n")
            sys.exit(1)

        except Exception as error:
            click.echo(f"{RED}Error:{RESET} {error}", err=True)
            sys.exit(1)

        else:
            return content, response_time, response


def display_debug_information(prompt, model, temperature):
    """
    Displays debug information.
    """
    click.echo("---\n" + click.style("Debug information:", fg="yellow"), err=True)
    click.echo(err=True)

    click.echo(click.style("Prompt:", fg="red"), nl=False, err=True)
    for role in prompt:
        click.echo(err=True)
        click.echo(click.style(f"{role['role']}:", fg="blue"), err=True)
        click.echo(f"{role}", err=True)
    click.echo(err=True)

    click.echo(click.style("Model:", fg="red"), err=True)
    click.echo(f"{model=}", err=True)
    click.echo(err=True)

    click.echo(click.style("Temperature:", fg="red"), err=True)
    click.echo(f"{temperature=}", err=True)
    click.echo(err=True)

    click.echo(click.style("End of debug information.", fg="yellow"), err=True)
    click.echo("---\n", err=True)


def display_tokens_count_and_cost(prompt, model):
    """
    Displays the number of tokens in the prompt and the cost of the prompt.
    """
    full_prompt = prompt[0]["content"] + prompt[1]["content"]
    number_of_tokens = openai_utils.num_tokens_from_string(full_prompt, model)
    cost = openai_utils.estimate_prompt_cost(prompt)[model]

    click.echo(
        "Number of tokens in the prompt:"
        f" {click.style(str(number_of_tokens), fg='yellow')}."
    )
    click.echo(
        f"Cost of the prompt for the {click.style(model, fg='blue')} model is:"
        f" {click.style(f'${cost}', fg='yellow')}."
    )
    click.echo(
        "Please note that this cost applies only to the prompt, not the"
        " subsequent response."
    )
    sys.exit(0)


def get_api_key() -> str:
    """
    Return the OpenAI API key.
    """
    key_file_path = get_api_key_path()
    with open(key_file_path, "r", encoding="UTF-8") as key_file:
        return key_file.read().strip()


def get_api_key_path() -> Path:
    """
    Return the path to the keys file.
    """
    key_file_path = Path.home() / ".config" / "lmt" / "key.env"
    if not key_file_path.exists():
        key_file_path.parent.mkdir(parents=True, exist_ok=True)
        key_file_path.touch()
    return key_file_path


def write_key(key: str) -> None:
    """
    Write the OpenAI API key to the key file.
    """
    key_file_path = get_api_key_path()
    with open(key_file_path, "w", encoding="UTF-8") as key_file:
        key_file.write(key)


def set_key() -> None:
    """
    Add the OpenAI API key.
    """
    key_path = get_api_key_path()
    key = get_api_key()
    if key:
        click.echo(click.style("Error: ", fg="red") + "API key already exists.")
        click.echo(f"Use `{click.style('lmt key edit', fg='blue')}` to edit it.")
        return

    key = click.prompt("Your OpenAI API key", hide_input=True)
    write_key(key)
    click.echo(f"{click.style('Success!', fg='green')} API key added.")
    click.echo(f"\nThe API key is stored in {key_path}.")


def edit_key() -> None:
    """
    Edit the OpenAI API key.
    """
    key_file_path = get_api_key_path()
    key = get_api_key()
    if not key:
        click.echo(click.style("Error: ", fg="red") + "API key does not exist.")
        click.echo("You will now be prompted to add it.\n")
        set_key()
        return

    original_key = key
    new_key = click.prompt("Your OpenAI API key", hide_input=True)
    if original_key != new_key:
        write_key(new_key)
        click.echo(f"{click.style('Success!', fg='green')} API key was updated.")
    else:
        click.echo("No changes were made.")
    click.echo(f"\nThe API key is stored in {key_file_path}.")
