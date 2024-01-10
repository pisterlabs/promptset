# imports
import re
import os
import toml
import typer
import openai
import marvin
import random

import logging as log

from marvin import ai_fn, ai_model, ai_classifier, AIApplication
from marvin.tools import tool
from marvin.prompts.library import System, User, ChainOfThought
from marvin.engine.language_models import chat_llm

from typing import Optional

from rich.console import Console
from dotenv import load_dotenv

## local imports
from dkdc.utils import dkdconsole

# setup console
console = Console()

# setup AI
model = "azure_openai/gpt-4-32k"
marvin.settings.llm_model = model
model = chat_llm(model)


# testing
def testing_run():
    console.print(f"testing.ai: ", style="bold violet", end="")
    console.print(f"done...")
