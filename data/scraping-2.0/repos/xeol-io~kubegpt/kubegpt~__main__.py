# __main__.py
import os
from kubegpt.prompt import prompt
import openai
import argparse
import warnings


def setup_cli():
    parser = argparse.ArgumentParser(description="kubegpt")
    parser.add_argument("prompt", help="The question to ask your Kubernetes cluster")
    parser.add_argument(
        "--version",
        action="version",
        version="0.0.7",
        help="Print the version and exit",
    )
    return parser.parse_args()


def main():
    # supress the warning "The shell tool has no safeguards by default"
    warnings.filterwarnings("ignore")

    args = setup_cli()

    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "Please set your OpenAI API key in the environment variable OPENAI_API_KEY"
        )

    openai.api_key = os.getenv("OPENAI_API_KEY")

    if not args.prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    prompt(args.prompt)
