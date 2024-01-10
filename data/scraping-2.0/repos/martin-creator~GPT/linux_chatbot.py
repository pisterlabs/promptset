import os
import openai
import click

# develop a command-line tool that can assist us with Linux commands through conversation.
# Click documentation: https://click.palletsprojects.com/en/8.1.x/


def init_api():
    ''' Load API key from .env file'''
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value

    openai.api_key = os.environ["API_KEY"]
    openai.organization = os.environ["ORG_ID"]


init_api()

# content stuffing the prompt
_prompt = """
Input: List all the files in the current directory
Output: ls -l

Input: List all the files in the current directory, including hidden files
Output: ls -la

Input: Delete all the files in the current directory
Output: rm *

Input: Count the number of occurrences of the word "sun" in the file "test.txt"
Output: grep -o "sun" test.txt | wc -l

Input:{}
Output:
"""


while True:
    request = input(click.style("Input (type 'exit' to quit): ", fg="green"))
    if request == "exit":
        break

    prompt = _prompt.format(request)

    try:
        result = openai.Completion.create(
            model="text-davinci-002",
            prompt=prompt,
            max_tokens=50,  # Adjust the max_tokens value as needed
            temperature=0.0,
        )

        command = result.choices[0].text.strip()

        if command == "":
            click.echo(click.style("No command generated.", fg="red"))
            continue

        click.echo( click.style("\nOutput: " + command, fg="yellow"))

        choice = input(click.style("Execute? (y/n): ", fg="green"))

        if choice == "y":
            os.system(command)
        elif choice == "n":
            continue
        else:
            click.echo( click.style("Invalid choice. Please enter 'y' or 'n'. ", fg="red"))

    except Exception as e:
        click.echo( click.style("The command could not be executed. {}".format(e), fg="red"))
        pass

    click.echo()
