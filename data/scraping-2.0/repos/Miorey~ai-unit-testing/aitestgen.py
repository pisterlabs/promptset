import os
import click
import json
from pathlib import Path
from openai import OpenAI
client = OpenAI()

@click.command()
@click.argument('filepath', type=click.Path(exists=True))
@click.option('--open-ai-env-var', required=False)
def generate_test(filepath: str, open_ai_env_var: str | None):
    # AI-TEST: don't test this function

    setup = validate_inputs(filepath, open_ai_env_var)
    execute_test_cover(setup)


def validate_inputs(filepath: str, open_ai_env_var: str | None) -> dict:
    open_api_key = os.getenv(open_ai_env_var or "OPENAI_API_KEY")
    if not open_api_key:
        raise click.ClickException('Api key env var is not set')
    client.api_key = open_api_key
    if not filepath:
        raise click.ClickException('File is not set')
    file_path = Path(filepath)

    # AI-TEST: for the test the valid existing file path is: './tests/valid_file.json'
    if not file_path.exists():
        raise click.ClickException(f'File {filepath} not exists')

    if file_path.suffix != ".json":
        raise click.ClickException("File does not have a .json extension.")

    try:
        with file_path.open('r') as f:
            setup = json.load(f)
            return setup
    except (IOError, json.JSONDecodeError) as e:
        raise click.ClickException(f"Unable to read json file {filepath}")


def execute_test_cover(gen_setup: dict[str, str | list[dict[str, str]]]):
    language = gen_setup["language"]
    for to_cover in gen_setup["files"]:
        # AI-TEST: Mock the open usage to avoid FileNotFound exception
        with open(to_cover["code"], 'r') as file:
            content = file.read()
        additional_comments: list[str] = gen_setup.get("additional-comments", [])

        # AI-TEST: mock the following openai call
        completion = client.chat.completions.create(
            model=gen_setup["model"],
            messages=[
                {"role": "system", "content": f"You are a f{language} developer"},
                {
                    "role": "user", "content": f"""
                    Write the unit test of the following code. 
                    The test should follow those rules:
                     - THE TEST SHOULD COVER 100% of the code.
                     - In the imports take in account that the test is in {to_cover["code"]} and the test in {to_cover["test"]}.
                     - The comments starting with `AI-TEST:` take them in consideration. 
                     - {', '.join(additional_comments)}
                     - BE SURE YOU USE ONLY THAT YOU WELL `import` the requirements.
                     - The test should be simple and with a cyclomatic complexity as lower as possible.
                     
                    Your answer should contain:
                     - NO SYNTAX HIGHLIGHTING.
                     - no introduction or explanation.
                     - ALL the test should be in the same snippet.
                    ```
                        {content}
                    ```
                    """
                }
            ]
        )
        ai_response = completion.choices[0].message.content
        only_code = ai_response.replace("```", "")
        with open(to_cover["test"], 'w') as f:
            f.write(only_code)


# AI-TEST: don't test this condition
if __name__ == '__main__':
    generate_test()
