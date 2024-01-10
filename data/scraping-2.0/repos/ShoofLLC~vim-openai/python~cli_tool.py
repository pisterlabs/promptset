# This code creates a CLI tool for OpenAI Codex using click. 
# The CLI accepts user input and it outputs the response to stdout.
# You can use the following code to start integrating your current prompt and settings into your application.
"""
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="davinci-codex",
  prompt="", # This is where the user input goes
  temperature=0,
  max_tokens=171,
  top_p=1,
  frequency_penalty=0.28,
  presence_penalty=0
)
"""
# Imports and constants
import click
import openai
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Functions
def get_response(prompt, engine="code-davinci-001", temperature=0, max_tokens=171, top_p=1, frequency_penalty=0.28, presence_penalty=0):
    """
    This function calls the OpenAI API to get a response.
    """
    openai.api_key = OPENAI_API_KEY

    response = openai.Completion.create(
        engine=engine,
        prompt=prompt, # This is where the user input goes
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )

    return response["choices"][0]["text"]


# CLI commands and groups
@click.group()
def cli():
    """This is a command line tool for OpenAI."""

    pass


@cli.command()
@click.argument("prompt")
@click.option("--engine", default="code-davinci-001", help="The engine to use.")
@click.option("--temperature", default=0, help="The temperature to use.")
@click.option("--max-tokens", default=171, help="The max tokens to use.")
@click.option("--top-p", default=1, help="The top p to use.")
@click.option("--frequency-penalty", default=0.28, help="The frequency penalty to use.")
@click.option("--presence-penalty", default=0, help="The presence penalty to use.")
def get(prompt, engine, temperature, max_tokens, top_p, frequency_penalty, presence_penalty):
    """This function calls the OpenAI API to get a response."""

    response = get_response(prompt=prompt, engine=engine, temperature=temperature, max_tokens=max_tokens, top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)

    print(response)


if __name__ == "__main__":
    cli()
