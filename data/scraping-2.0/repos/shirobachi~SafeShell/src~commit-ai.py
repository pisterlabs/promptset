#!/usr/bin/env python3
import sys
import os
import openai
from rich.console import Console
from rich.prompt import Prompt

console = Console()


def load():
    openai.api_key = os.getenv("OPENAI_API_KEY")


def get_diff():
    diff = os.popen("git diff --cached").read()

    if not diff:
        console.log("No changes to commit or not in git repo.")
        sys.exit(1)

    return diff

# get suggestions


def get_suggestions(diff, model="gpt-4"):
    if model == "static":
        return "Update roles in i3_arch_setup and managed_by_bare_repo tasks"

    return openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "I want you to act as the author of a commit message in git. I'll enter a git diff, and your job is to convert it into a useful commit message. Do not preface the commit with anything, use the present tense, return the full sentence. Write only title, max 72 characters."
            },
            {
                "role": "user",
                "content": diff
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    ).choices[0].message.content.capitalize()


def commit(message):
    os.system(f"git commit -m '{message}'")


def copy(message):
    os.system(f"echo '{message}' | xclip -sel clip")


def generate_commit_message(diff):
    console.log("Generating suggestions...")
    message = get_suggestions(diff)
    console.log(f"Suggestions: [bold]{message}[/bold]")

    c = Prompt.ask("Options: [underline2]r[/underline2]etry, [underline2]c[/underline2]ommmit, cop[underline2]y[/underline2]",
                   choices=['r', 'c', 'y'], default="c")

    if c == "r":
        return generate_commit_message()
    elif c == "c":
        return commit(message)
    elif c == "y":
        return copy(message)


if __name__ == "__main__":
    load()
    generate_commit_message(get_diff())
