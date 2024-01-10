import click
import openai
import os
import dotenv

from cli.questions import get_algorithms

dotenv.load_dotenv()


@click.command()
def quiz():
    """Generate a quiz about a specific algorithm or data structure"""
    openai.api_key = os.getenv("OPENAI_API_KEY")

    algos = get_algorithms()

    click.Choice(
        "What algorithm would you like to learn about?", algos["title"].tolist()
    )

    # chat_completion = openai.ChatCompletion.create(
    #     model="gpt-4-0613",
    #     messages=[{"role": "user", "content": "What is a binary search tree?"}],
    # )

    # print(chat_completion.choices[0].message.content)


if __name__ == "__main__":
    quiz()
