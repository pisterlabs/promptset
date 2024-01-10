from datetime import date
import click
import sys

sys.path.append(".")

from think.llm.openai import ChatGPT  # noqa E402
from think.chat import Chat  # noqa E402
from think.tool import tool  # noqa E402


@tool
def current_date() -> str:
    """
    Get the current date.

    :returns: current date in YYYY-MM-DD format
    """
    return date.today().isoformat()


@click.command()
@click.option("--api-key", "-k", default=None)
def main(api_key=None):
    """
    Ask GPT-4 how old it is, providing the current date as a tool.

    API key, if not provided, will be read from OPENAI_API_KEY environment variable.
    """
    llm = ChatGPT()
    chat = Chat("You are a helpful assistant.")
    chat.user("How old are you (in days since your knowledge cutoff)?")
    print(llm(chat, tools=[current_date]))


if __name__ == "__main__":
    main()
