#!/usr/bin/env python

import os
import sys
import click
from openai import OpenAI
from pathlib import Path
from rich.console import Console

default_model = 'gpt-3.5-turbo-instruct'
basedir = Path.home() / ".gpp"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or (basedir / "openai-key.txt").read_text()[:-1])
console = Console()

@click.command()
@click.argument('prompt', nargs=-1)
def main(prompt):
  """
  The instruct command ...
  """
  if len(prompt) == 0:
    # read the prompt from stdin
    prompt = [sys.stdin.read()]

  response = client.completions.create(
     model=default_model,
    prompt=prompt,
  )
  console.print(response['choices'][0]['text'])
  print(response)

if __name__ == '__main__':
    main()