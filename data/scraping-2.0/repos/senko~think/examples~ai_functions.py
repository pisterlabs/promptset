import sys
import click

sys.path.append(".")

from think.llm.openai import ChatGPT  # noqa E402
from think.ai import ai  # noqa E402


@ai
def haiku(topic: str) -> str:
    """
    Write a haiku about {{ topic }}
    """


@click.command()
@click.option("--api-key", "-k", default=None)
@click.argument("topic")
def main(topic, api_key=None):
    """
    Write and output a haiku about a given TOPIC using GPT-4.

    API key, if not provided, will be read from OPENAI_API_KEY environment variable.
    """
    llm = ChatGPT(api_key=api_key)
    print(haiku(llm, topic=topic))


if __name__ == "__main__":
    main()
