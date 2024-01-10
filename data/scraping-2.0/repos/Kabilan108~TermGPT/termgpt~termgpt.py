"""
TermGPT
-------

Your in-terminal LLM assistant.
"""

from io import StringIO
import os

from rich.console import Console
import openai as ai
import pydoc
import click


ai.api_key = os.environ["OPENAI_API_KEY"]
console = Console(file=StringIO(), force_terminal=True)


class TermGPT:
    """
    TermGPT
    -------

    Your in-terminal LLM assistant.

    Methods
    -------
    cli()
        The main CLI command.
    ask(question: str)
        Ask TermGPT a question.
    """

    system_prompt = [
        {
            "role": "system",
            "content": """
You are an assistant for a machine learning engineer.
You are tasked with helping the engineer with their work.
Your goal is to provide concise and accurate feedback to help the user solve their problem. 
Your response should include suggestions for the best way to solve the problem. 
These suggestions can be in the form of a list or a single answer. 
You should try your best to give correct solutions. 
If your response includes code snippets, make sure to format them correctly: 
one-liners should be wrapped in ` tags and longer snippets should be wrapped between ``` tags.
Make your responses friendly and helpful.
Code snippets should also be wrapped in [cyan]`[/cyan] or [cyan]```[/cyan] tags.
Emphasize important words or phrases with [bold]bold[/bold] or [italic]italic[/italic] tags.
You can also use [magenta]...[/magenta] for additional emphasis.
Use [green]...[/green] to for titles or headings.
You are allowed to use emojis using the format :emoji_name:
            """,
        }
    ]

    @staticmethod
    def _generate_content(response):
        """
        Generate content from the response.
        """
        lines = [x for x in response.splitlines() if x]
        for line in lines:
            console.print(line)

    @click.group()
    @staticmethod
    def cli() -> None:
        """
        Welcome to TermGPT! I am your in-terminal LLM assistant.

        To get started, type `termgpt ask` and then your question.
        """
        return

    @cli.command()
    @click.argument("question", type=str, required=True)
    @staticmethod
    def ask(question) -> None:
        """
        Ask TermGPT a question.
        """

        response = ai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=TermGPT.system_prompt + [{"role": "user", "content": question}],
        )

        response = response["choices"][0]["message"]["content"]  # type: ignore
        TermGPT._generate_content(response)

        pydoc.pager(console.file.getvalue())  # type: ignore
        console.file.close()  # type: ignore


if __name__ == "__main__":
    TermGPT.cli()
