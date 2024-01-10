import openai
import os
import typer
import nltk

app = typer.Typer()


def _generate_prompt(refactor_or_edit_instructions, code, language):
    return f"""
    {'Refactor' if 'refactor' in refactor_or_edit_instructions.lower() else 'Edit'} the following {language} code: {refactor_or_edit_instructions}
    
    Please provide an extremely succinct human explanation of the changes made to the code
    and return the edited code in a new file, delimited by ''. Don't use '' other than
    between the sections, and don't add space between sections either.
    
    Take liberties to fix technical problems if you find them, but make sure to explain it clearly in comments
    and the explanation section, and include line numbers in the explanation section if you do.
    
    Ensure that code is well documented and formatted.
    {" Use google docstrings and black formatting."if language == "python" else ""}
    
    {code}""".strip()


def _refactor_or_edit(
    file_path: str,
    refactor_or_edit_instructions: str,
    explanation_file: str = None,
    model: str = "text-davinci-003",
    language: str = "python",
    debug: bool = False,
):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # open the file and escape the code as a code block
    with open(file_path, "r") as file:
        code = f"```{language}\n" + file.read() + "\n```"

    # specify the prompt
    prompt = _generate_prompt(refactor_or_edit_instructions, code, language)
    tokens = nltk.word_tokenize(prompt)

    #! Yeah this math is BS, closeish though...
    max_tokens = (round(4097 - (7 / 4) * len(tokens)),)

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

    explanation = refactored_code.split("")[0]
    refactored_code = "".join(refactored_code.split("")[1:])
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
    _refactor_or_edit(
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
    _refactor_or_edit(
        file_path, edit_instructions, explanation_file, model, language, debug
    )


@app.command()
def configure():
    """
    Configure the OpenAI secret key for the codegpt CLI.

    If the OPENAI_SECRET_KEY is already set in the environment variables, this command will not do anything.
    Otherwise, it will prompt the user for the secret key and create a .env file with the secret key.
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
