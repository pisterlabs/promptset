import click
import openai
import json
import configparser
import os
import subprocess
import sys


script_dir = os.path.dirname(os.path.realpath(__file__))
config_file = os.path.join(script_dir, 'config.ini')


def read_api_key():
    config = configparser.ConfigParser()
    config.read(config_file)
    try:
        return config.get('openai', 'api_key')
    except (configparser.NoSectionError, configparser.NoOptionError):
        return None


def write_api_key(api_key):
    config = configparser.ConfigParser()
    config.read(config_file)
    if not config.has_section('openai'):
        config.add_section('openai')
    config.set('openai', 'api_key', api_key)

    with open(config_file, 'w') as f:
        config.write(f)


def update_api_key():
    print("API key is invalid. Please enter a new OpenAI API key.")
    OPENAI_API_KEY = input("API key: ").strip()
    write_api_key(OPENAI_API_KEY)
    openai.api_key = OPENAI_API_KEY


def prepare_api_key():
    api_key = read_api_key()

    if api_key:
        openai.api_key = api_key
    else:
        update_api_key()


def get_user_input():
    try:
        return click.prompt(click.style("You", fg='green', bold=True))
    except click.exceptions.Abort:
        raise KeyboardInterrupt


def create_chat_params(model, messages, options):
    params = {
        "model": model,
        "messages": messages,
    }

    for key, value in options.items():
        if value is not None:
            params[key] = value

    return params


def print_assistant_response(response):
    click.echo(click.style("Assistant:", fg='yellow', bold=True))

    assistant_response = ""
    for chunk in response:
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            chunk_text = delta["content"]
        if chunk_text:
            assistant_response += chunk_text
            click.echo(click.style(f"{chunk_text}", fg='yellow'), nl=False)

    click.echo("")  # Add a newline at the end
    return assistant_response.strip()


@click.command()
@click.option("--model", default="gpt-4", help="The model to use.")
@click.option("--system_message",
              default="You are a cli chat bot using OpenAI's API.",
              help="A custom system message.")
@click.option("--temperature", default=None, type=float,
              help="The sampling temperature.")
@click.option("--top_p", default=None, type=float,
              help="The nucleus sampling value.")
@click.option("--n", default=None, type=int,
              help="The number of chat completion choices.")
@click.option("--stream", default=True, type=bool,
              help="Enable partial message deltas streaming.")
@click.option("--stop", default=None, type=str,
              help="The stop sequence(s) for the API.")
@click.option("--max_tokens", default=None, type=int,
              help="The maximum number of tokens to generate.")
@click.option("--presence_penalty", default=None, type=float,
              help="The presence penalty.")
@click.option("--frequency_penalty", default=None, type=float,
              help="The frequency penalty.")
@click.option("--logit_bias", default=None, type=str,
              help="The logit bias as a JSON string.")
@click.option("--user", default=None, type=str,
              help="A unique identifier for the end-user.")
def start_chat(model, system_message, temperature, top_p,
               n, stream, stop, max_tokens, presence_penalty,
               frequency_penalty, logit_bias, user):
    click.echo(click.style(f"System: {system_message}",
                           fg='yellow',
                           bold=True))

    messages = [{"role": "system", "content": system_message}]

    options = {
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "stream": stream,
        "stop": stop,
        "max_tokens": max_tokens,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "logit_bias": json.loads(logit_bias) if logit_bias else None,
        "user": user
    }

    while True:
        try:
            user_input = get_user_input()
            messages.append({"role": "user", "content": user_input})

            params = create_chat_params(model, messages, options)

            try:
                response = openai.ChatCompletion.create(**params)
            except openai.error.APIConnectionError as e:
                click.echo("The server could not be reached")
                click.echo(e.__cause__)  # an underlying Exception, likely raised within httpx.
                messages.pop()
                continue
            except openai.error.RateLimitError as e:
                click.echo("A 429 status code was received; we should back off a bit.")
                messages.pop()
                continue
            except openai.error.APIStatusError as e:
                click.echo("Another non-200-range status code was received")
                click.echo(e.status_code)
                click.echo(e.response)
                messages.pop()
                continue
            except openai.error.AuthenticationError:
                update_api_key()
                messages.pop()
                continue

            assistant_response = print_assistant_response(response)
            messages.append({
                "role": "assistant",
                "content": assistant_response
                })

        except (KeyboardInterrupt, EOFError):
            click.echo("\nExiting the chat session.")
            end_program()
            break


def end_program():
    if original_title:
        change_terminal_window_name(original_title)
    else:
        reset_tmux_title()


def change_terminal_window_name(title):
    sys.stdout.write(f"\x1b]2;{title}\x07")
    #os.system(title)
    os.system(f'tmux rename-window {title}')


def reset_tmux_title():
    try:
        subprocess.check_call(['tmux', 'setw', 'automatic-rename', 'on'])
    except subprocess.CalledProcessError as e:
        print('An error occurred while resetting tmux window title')
        print(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}")


if __name__ == "__main__":
    global original_title
    original_title = os.popen('xdotool getactivewindow getwindowname').read().strip()
    change_terminal_window_name("ChatGPT")
    prepare_api_key()
    start_chat()
