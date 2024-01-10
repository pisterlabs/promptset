"""
This module implements a CLI that can be used to interact with the CoD
summary generator.
"""
import os
import json
import click
import openai
import cod


@click.group()
@click.option(
    "opeanai_key",
    "--openai_key",
    default="",
    show_default=False,
    help="OpenAI API key.",
)
def init_openai(openai_key: str):
    """
    Initialize the OpenAI API.
    """
    if openai_key:
        openai.api_key = openai_key
    else:
        openai.api_key = os.environ["OPENAI_API_KEY"]
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY environment variable must be set")


@click.group()
@click.version_option(version="1.0.0")  # Set the version number
def cli():
    """Version CLI"""


@cli.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option(
    "steps",
    "--steps",
    default=5,
    show_default=True,
    type=click.IntRange(min=1, max=5),
    help="Number of steps.",
)
@click.option("debug", "--debug", default=False, show_default=True)
def summarize(filename: str, steps: int, debug: bool) -> list:
    """
    Generate a summary of a document using the CoD prompt.
    """
    openai.api_key = os.environ["OPENAI_API_KEY"]
    # read the file
    with open(filename, "r", encoding="utf-8") as file:
        file_content = file.read()
    # generate the summary
    summary = cod.cod_summarize(file_content, steps, debug)
    # print the last summary
    last_summary = summary[-1]
    # Get JSON response from the last_summary
    # convert the string to json
    json_summary = json.loads(last_summary)
    print(json_summary["summary"])
    return summary


if __name__ == "__main__":
    cli()
