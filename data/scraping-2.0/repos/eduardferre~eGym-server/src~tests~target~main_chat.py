import openai
import typer
import config

from rich import print
from rich.table import Table


def main():
    openai.api_key = "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"

    print("[bold green]ChatGPT API with Python[/bold green]")

    table = Table("Command", "Description")
    table.add_row("exit", "Exit from application")
    table.add_row("new", "Create a new conversation")

    print(table)

    ctxt = typer.prompt("\nSpecify the context... ")

    context = {"role": "system", "content": ctxt}  # it defines a context
    messages = [context]

    while True:
        content = __prompt()

        if content == "new":
            print("New conversation has been created")

            ctxt = typer.prompt("\nSpecify the context... ")

            context = {"role": "system", "content": ctxt}  # it defines a context
            messages = [context]
            content = __prompt()

        messages.append({"role": "user", "content": content})

        response = openai.ChatCompletion.create(
            model="text-davinci-003", messages=messages
        )

        response_content = response.choices[0].message.content

        messages.append({"role": "assistant", "content": response_content})

        print(f"[bold green]> [/bold green][green]{response_content}[/green]")


def __prompt() -> str:
    prompt = typer.prompt("\nQuestion... ")

    if prompt == "exit":
        exit = typer.confirm("Are you sure?")
        if exit:
            print("Bye bye")
            raise typer.Abort()

        return __prompt()

    return prompt


if __name__ == "__main__":
    typer.run(main)
