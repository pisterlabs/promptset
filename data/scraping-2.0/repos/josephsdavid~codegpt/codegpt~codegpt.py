import openai
import os
import typer
import nltk

app = typer.Typer()


def generate_prompt(refactor_or_edit_instructions, code, language):
    """Generate a prompt from the given instructions and code.

    Args:
        refactor_or_edit_instructions (str): Instructions for refactoring or editing the code.
        code (str): The code to be refactored or edited.
        language (str): The language of the code.

    Returns:
        str: The generated prompt.
    """
    return f"""
    {'Refactor' if 'refactor' in refactor_or_edit_instructions.lower() else 'Edit'} the following {language} code: {refactor_or_edit_instructions}

    Please provide an extremely succinct human explanation of the changes made to the code
    and return the edited code in a new section, delimited by '==='. Don't use '===' other than
    between the sections (don't remove it if it's present though!), and don't add space between sections.

    Ensure that code is well documented and formatted.
    {" Use google docstrings and black formatting."if language == "python" else ""}

    You must explain what you did, even if you don't make a change.

    Code:
    {code}""".strip()


def refactor_or_edit(
    file_path: str,
    refactor_or_edit_instructions: str,
    explanation_file: str = None,
    model: str = "text-davinci-003",
    language: str = "python",
    debug: bool = False,
):
    """Refactor or edit the given file.

    Args:
        file_path (str): The path to the file to be refactored or edited.
        refactor_or_edit_instructions (str): Instructions for refactoring or editing the code.
        explanation_file (str, optional): The path to the file to save the explanation. Defaults to None.
        model (str, optional): GPT-3 model to use. Defaults to "text-davinci-003".
        language (str, optional): The language of the code. Defaults to "python".
        debug (bool, optional): If True, save the response from the model in a JSON file. Defaults to False.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # open the file and escape the code as a code block
    with open(file_path, "r") as file:
        orig = file.read()
        code = f"```{language}\n" + orig + "\n```"
        with open(f"{file_path}.bak", 'w') as backup:
            backup.write(orig)



    # specify the prompt
    prompt = generate_prompt(refactor_or_edit_instructions, code, language)
    tokens = nltk.word_tokenize(prompt)

    #! Yeah this math is BS, closeish though...
    max_tokens = round(4097 - (7 / 4) * len(tokens))

    typer.confirm(
        f"This prompt is {len(tokens)} tokens, are you sure you want to continue?\nThe most GPT-3 can return in response is {max_tokens}.",
        default=True,
        abort=True,
    )

    # send the prompt to the model
    response = openai.Completion.create(
        max_tokens=max_tokens,
        engine=model,
        prompt=prompt,
        n=1,
        stop=None,
        temperature=0.6,
    )

    if debug:
        import json

        # write the response
        with open(file_path + ".resp.json", "w") as file:
            file.write(json.dumps(response))

    # print the response from the model
    refactored_code = response["choices"][0]["text"]

    explanation = refactored_code.split("===")[0]
    refactored_code = "".join(refactored_code.split("===")[1:])
    print(explanation)

    old_file_path = file_path + ".old"
    os.rename(file_path, old_file_path)

    # write the refactored code to the original file
    with open(file_path, "w") as file:
        file.write(refactored_code)

    # write the refactored code to the original file
    with open(
        explanation_file if explanation_file else file_path + ".explained.txt", "w"
    ) as file:
        file.write(explanation)


@app.command()
def refactor(
    file_path: str,
    refactor_instructions: str,
    explanation_file: str = None,
    model: str = "text-davinci-003",
    language: str = "python",
    debug: bool = False,
):
    """Refactor the given file according to the given instructions.

    Args:
        file_path (str): The path to the file to be refactored.
        refactor_instructions (str): Instructions for refactoring the code.
        explanation_file (str, optional): The path to the file to save the explanation. Defaults to None.
        model (str, optional): GPT-3 model to use. Defaults to "text-davinci-003".
        language (str, optional): The language of the code. Defaults to "python".
        debug (bool, optional): If True, save the response from the model in a JSON file. Defaults to False.
    """
    refactor_or_edit(
        file_path, refactor_instructions, explanation_file, model, language, debug
    )


@app.command()
def varnames(
    file_path: str,
    refactor_instructions: str = "In the following code, rename variables as you see appropriate for it to be easier to read. Don't touch any of the code otherwise, other than to update comments.",
    explanation_file: str = None,
    model: str = "text-davinci-003",
    language: str = "python",
    debug: bool = False,
):
    """Refactor the given file to rename variables as appropriate.

    Args:
        file_path (str): The path to the file to be refactored.
        refactor_instructions (str): Instructions for refactoring the code.
        explanation_file (str, optional): The path to the file to save the explanation. Defaults to None.
        model (str, optional): GPT-3 model to use. Defaults to "text-davinci-003".
        language (str, optional): The language of the code. Defaults to "python".
        debug (bool, optional): If True, save the response from the model in a JSON file. Defaults to False.
    """
    refactor_or_edit(
        file_path, refactor_instructions, explanation_file, model, language, debug
    )


@app.command()
def comment(
    file_path: str,
    refactor_instructions: str = "In the following code, make no code changes but add comments. Keep them succinct, but explain everything you can if it's helpful. Add function or class strings where you can.",
    explanation_file: str = None,
    model: str = "text-davinci-003",
    language: str = "python",
    debug: bool = False,
):
    """Edit the given file to add comments."""
    refactor_or_edit(
        file_path, refactor_instructions, explanation_file, model, language, debug
    )


@app.command()
def edit(
    file_path: str,
    edit_instructions: str,
    explanation_file: str = None,
    model: str = "text-davinci-003",
    language: str = "md",
    debug: bool = False,
):
    """A Generic edit option, meant for editing markdown blog posts. Basically refactor with some extra instructions."""
    refactor_or_edit(
        file_path, edit_instructions, explanation_file, model, language, debug
    )


@app.command()
def configure():
    """
    Configure the OpenAI secret key for the codegpt CLI.
    """
    # check if the secret key is already set in the environment variables
    if "OPENAI_SECRET_KEY" in os.environ:
        print("The OPENAI_SECRET_KEY is already set in the environment variables.")
        return
    else:
        typer.confirm(
            """
We no longer handle this, it felt a little iffy security wise. I recommend setting your API key as an environment variable:
https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety

Windows users can also use `setx` like:

`$ setx OPENAI_SECRET_KEY=<YOUR_API_KEY>`

from an admin console.
        """.strip()
        )


if __name__ == "__main__":
    app()
