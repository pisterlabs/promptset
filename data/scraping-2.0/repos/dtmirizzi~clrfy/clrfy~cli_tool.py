import click
from .llm import LLM
from .langchain import LangChain

@click.command()
@click.argument('input', type=click.File('r'))
def cli(input):
    """This is a CLI tool that corrects grammar in string blocks and comments."""
    text = input.read()

    # Initialize the LLM and LangChain
    llm = LLM()
    langchain = LangChain()

    # Correct the grammar
    corrected_text = langchain.correct(llm.process(text))

    click.echo(corrected_text)

if __name__ == "__main__":
    cli()