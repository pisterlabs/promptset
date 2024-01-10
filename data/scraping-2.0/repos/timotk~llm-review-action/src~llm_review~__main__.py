import os
import string
from pathlib import Path
from typing import Optional

import typer
from openai import AzureOpenAI
from pydantic import ValidationError

from llm_review.models import Comment, Comments
from llm_review.prompt import create_prompt

AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
AZURE_OPENAI_API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"]

client = AzureOpenAI(
    api_version=AZURE_OPENAI_API_VERSION,
)


PROMPT = string.Template(
    """
Please review my text. For each line, make a suggestion.
Consider writing style, conciseness and writing style.
If you do, output it according to the following JSON schema. Only output raw JSON. Do not write anything else.:
```shell
::notice file={$filepath},line={lineno},col=1::{comment}
```
Where `lineno` is the line number, and `comment` is what you want to suggest. Replace those values.

Here is the text to review:
"""
)


def comment_as_github_annotation(comment: Comment, file: Path) -> str:
    """
    Create a github notice string from a comment
    By printing the resulting value during the Github action run, the comment will added as an annotation to the file
    See the docs for more details https://docs.github.com/en/actions/using-workflows/workflow-commands-for-github-actions#setting-a-notice-message
    """
    s = f"::notice file={file},line={comment.line_start}"
    if comment.line_end:
        s += f",endLine={comment.line_end}"
    s += f"::{comment.content}"
    return s


def files_from_dir(dir: Path) -> list[Path]:
    files = [file for file in dir.glob("**/*") if file.is_file()]
    return files


def files_from_paths(paths: list[Path]) -> list[Path]:
    files = []
    for fp in paths:
        if fp.is_dir():
            files.extend(files_from_dir(fp))
        else:
            files.append(fp)
    return files


def query_llm(prompt: str) -> str | None:
    """Query the LLM with the given prompt."""
    # print(prompt)
    completion = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.0,
    )
    return completion.choices[0].message.content


def read_file(path: Path) -> Optional[str]:
    with path.open() as f:
        content = f.read()
        if len(content) == 0:
            print("File is empty! Ignoring.")
            return None
    return content


def review_content(content: str, user_instruction: Optional[str]) -> list[Comment]:
    """Review the given content and return a list of comments"""

    prompt = create_prompt(content, user_instruction=user_instruction)
    llm_output = query_llm(prompt)
    if not llm_output:
        return []

    # Strip backticks which sometimes are given by the llm
    llm_output = llm_output.lstrip("```json").rstrip("```")

    try:
        comments = Comments.model_validate_json(llm_output).root
    except ValidationError as e:
        print(f"LLM output did not validate against the schema: {e}")
        print(f"LLm output: {llm_output}")
        raise typer.Exit(1)
    return comments


def main(
    paths: list[Path] = typer.Argument(..., help="The file(s) to review"),
    user_instruction: Optional[str] = typer.Option(
        None, help="The additional user provided prompt"
    ),
    template_name: Optional[str] = typer.Option(
        None, help="The name of the template (from the templates dir)"
    ),
) -> None:
    if user_instruction and template_name:
        print("You can only specify one of --user-instruction or --template-name")
        raise typer.Exit(1)

    if template_name:
        template_path = Path("templates") / (template_name + ".txt")
        if not template_path.exists():
            print(f"Template {template_name} not found in templates directory")
            raise typer.Exit(1)
        print(f"Using template {template_path}")
        user_instruction = template_path.read_text()

    print(f"Using model {AZURE_OPENAI_DEPLOYMENT}")

    files = files_from_paths(paths)
    print(f"Got {len(files)} files")

    for file in files:
        print(f"Reviewing file {file}")
        content = read_file(file)
        if not content:
            # Empty file or file not found
            continue
        comments = review_content(content, user_instruction=user_instruction)
        for comment in comments:
            print(comment_as_github_annotation(comment, file))


if __name__ == "__main__":
    typer.run(main)
