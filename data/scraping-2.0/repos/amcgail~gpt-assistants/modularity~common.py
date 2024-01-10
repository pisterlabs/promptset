from openai import OpenAI

import time
import threading
from pathlib import Path
import os
import importlib, inspect
import json
import datetime as dt

BASE_DIR = os.path.dirname(__file__)

from dotenv import load_dotenv
dotenv_path_here = os.path.join(BASE_DIR, '.env')
if os.path.exists(dotenv_path_here):
    load_dotenv(dotenv_path=dotenv_path_here)
else:
    load_dotenv('.env')

client = OpenAI()

def indent(text, amount=4, ch=' '):
    lines = text.splitlines()
    padding = amount * ch
    return '\n'.join(padding + line for line in lines)

def flatten_whitespace(text):
    lines = text.splitlines()

    # first and last lines might be empty
    if not len(lines[0].strip()):
        lines = lines[1:]
    if not len(lines[-1].strip()):
        lines = lines[:-1]

    # calculate the number of spaces at the beginning of each line
    spaces = [len(line) - len(line.lstrip()) for line in lines if len(line.strip())]

    # get rid of min(spaces) spaces at the beginning of each line
    text = '\n'.join(line[min(spaces):] for line in lines)
    return text

from .modules import *
from .tools import *