import logging
import pathlib
from importlib import resources
from typing import Annotated, Optional

import openai
import questionary
import typer
from openai.types import chat
from rich import live, markdown

from ai_commit_cli import commit, token
from ai_commit_cli.external import bitwarden, git
from ai_commit_cli.external import pre_commit as pre


def main(
    api_key: Annotated[Optional[str], typer.Option(envvar="OPENAI_API_KEY")] = None,
    diff: Annotated[Optional[str], typer.Option()] = None,
    diff_file: Annotated[
        Optional[pathlib.Path], typer.Option(exists=True, dir_okay=False)
    ] = None,
    exclude: Annotated[Optional[list[str]], typer.Option("-e", "--exclude")] = None,
    include: Annotated[Optional[list[str]], typer.Option("-i", "--include")] = None,
    pre_commit: Annotated[bool, typer.Option()] = True,
    prompt: Annotated[Optional[str], typer.Option()] = None,
    prompt_file: Annotated[
        Optional[pathlib.Path], typer.Option(exists=True, dir_okay=False)
    ] = None,
    *,
    model: Annotated[
        str,
        typer.Option(
            help="ID of the model to use. See the model endpoint compatibility table "
            "for details on which models work with the Chat API."
        ),
    ] = "gpt-3.5-turbo-1106",
    max_tokens: Annotated[
        int,
        typer.Option(
            min=0,
            help="The maximum number of tokens to generate in the chat completion.",
        ),
    ] = 500,
    temperature: Annotated[
        float,
        typer.Option(
            help="What sampling temperature to use, between 0 and 2. Higher values "
            "like 0.8 will make the output more random, while lower values like 0.2 "
            "will make it more focused and deterministic.",
            min=0.0,
            max=2.0,
        ),
    ] = 0.0,
) -> None:
    if api_key is None:
        api_key = bitwarden.get_notes("OPENAI_API_KEY")
    if not exclude:
        exclude = ["*-lock.*", "*.lock"]
    if not include:
        include = []
    if prompt is None:
        if prompt_file is None:
            with resources.files("ai_commit_cli").joinpath(
                "docs", "prompt.md"
            ).open() as f:
                prompt = f.read()
        else:
            prompt = prompt_file.read_text()
    if pre_commit:
        pre.run()
    if diff is None:
        if diff_file is None:
            diff = git.diff(exclude=exclude, include=include)
            git.status(exclude=exclude, include=include)
        else:
            diff = diff_file.read_text()
    client: openai.OpenAI = openai.OpenAI(api_key=api_key)
    messages: list[chat.ChatCompletionMessageParam] = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": diff},
    ]
    logging.debug(messages)
    stream: openai.Stream[chat.ChatCompletionChunk] = client.chat.completions.create(
        messages=messages,
        model=model,
        stream=True,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    message: str = ""
    with live.Live() as live_panel:
        for chunk in stream:
            if chunk.choices[0].delta.content is None:
                message += "\n"
            else:
                message += chunk.choices[0].delta.content
                live_panel.update(markdown.Markdown(commit.sanitize(message)))
    try:
        num_tokens_prompt: int = token.num_tokens_from_string(prompt, model=model)
        num_tokens_diff: int = token.num_tokens_from_string(diff, model=model)
        num_tokens_input: int = token.num_tokens_from_messages(messages, model=model)
        logging.info(
            "Input Tokens: %d = %d (Prompt) + %d (Diff) + %d",
            num_tokens_input,
            num_tokens_prompt,
            num_tokens_diff,
            num_tokens_input - (num_tokens_prompt + num_tokens_diff),
        )
        num_tokens_output: int = token.num_tokens_from_string(message, model=model)
        logging.info("Output Tokens: %d", num_tokens_output)
        pricing_input: float
        pricing_output: float
        pricing_input, pricing_output = token.pricing(model)
        pricing_input *= num_tokens_input
        pricing_output *= num_tokens_output
        logging.info(
            "Pricing: $%f = $%f (Input) + $%f (Output)",
            pricing_input + pricing_output,
            pricing_input,
            pricing_output,
        )
    except NotImplementedError as e:
        logging.error(e)
    message = commit.sanitize(message)
    confirm: bool = questionary.confirm(
        message="Confirm the commit message?"
    ).unsafe_ask()
    if confirm:
        git.commit(message=message)
    else:
        print(message)
