# imports
import re
import os
import toml
import typer
import openai
import marvin

import logging as log

from marvin import ai_fn, ai_model
from rich import print
from rich.console import Console
from dotenv import load_dotenv

## local imports
from src.common import codai

# load .env file
load_dotenv()

# load config.toml
try:
    config = toml.load("config.toml")["test"]
except:
    config = {}

# configure logger
log.basicConfig(level=log.INFO)

# configure rich
console = Console()

# icode
def testing_run():
    codai(end="")
    console.print(f"\ndone...")
