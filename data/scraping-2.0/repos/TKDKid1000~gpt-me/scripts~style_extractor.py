import json
import os
import re
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from os import path
from random import sample, seed
from time import time

import openai
from dotenv import load_dotenv

from gptme.text_styler import TextStyler

load_dotenv(dotenv_path=".env")

openai.api_key = os.environ["OPENAI_KEY"]

parser = ArgumentParser(
    prog="GPT-me Style Extractor",
    description="Scans through a random sample of your messages to extract your approximate text style.",
)
parser.add_argument("filename")
parser.add_argument("-a", "--author", type=str, required=True)
parser.add_argument("-S", "--seed", type=int, default=time())
parser.add_argument("-m", "--min_length", type=int, default=20)
parser.add_argument("-s", "--samples", type=int, default=10)
parser.add_argument("-p", "--pretty", action=BooleanOptionalAction)

ARGS = parser.parse_args(sys.argv[1:])

with open(ARGS.filename, encoding="utf8") as transcript_file:
    transcript_lines = transcript_file.readlines()


transcript = []

for line in transcript_lines:
    author = re.match(r"^(\w+):", line)
    if author is not None and author[1] == ARGS.author:
        line = re.sub(r"^\w+: ", "", line)
        transcript.append(line)
    elif author is None:
        transcript[-1] += line

transcript = [line.strip() for line in transcript if len(line) >= ARGS.min_length]

seed(ARGS.seed)

message_samples = sample(transcript, ARGS.samples)

text_styler = TextStyler()

text_styler.extract_style(message_samples)

print(text_styler.style)

style_json = {"style": text_styler.style, "seed": ARGS.seed, "messages": message_samples}

output_path = f".styles/{time()}/{path.split(ARGS.filename)[1]}.json"

os.makedirs(path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf8") as f:
    json.dump(style_json, f, indent=4 if ARGS.pretty else None)
