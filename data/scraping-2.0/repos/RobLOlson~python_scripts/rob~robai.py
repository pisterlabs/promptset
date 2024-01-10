import argparse
import datetime
import os
import re
import shelve
import sys
from pathlib import Path

import openai
import rich

NOW = datetime.datetime.today()
openai.api_key = os.getenv("OPENAI_API_KEY")

USER = "Stranger"
FILE = Path(sys.argv[0])

os.makedirs(FILE.parent / "db", exist_ok=True)

# Create the parser
my_parser = argparse.ArgumentParser(
    prog=sys.argv[0],
    allow_abbrev=True,
    add_help=True,
    usage=f"{FILE.stem} [--user USER] [--clear] PROMPT",
    description="Use GPT-3 to complete a response",
    epilog="(C) Rob",
)

# Add the arguments
my_parser.add_argument(
    "Prompt",
    metavar="path",
    nargs="?",
    action="store",
    type=str,
    help="the prompt GPT-3 should reply to",
)

my_parser.add_argument(
    "-u",
    "--user",
    nargs="?",
    action="store",
    type=str,
    help="the person GPT-3 should reply to",
)

my_parser.add_argument(
    "-c",
    "--clear",
    action="store_true",
    help="Clear user's database of convos.",
)

my_parser.add_argument(
    "-i",
    "--inspect",
    action="store_true",
    help="Clear user's database of convos.",
)

# Execute the parse_args() method
args = my_parser.parse_args()

if args.clear and args.user:
    if input("Are you sure you want to clear the conversation database? [y/n]\n") in [
        "y",
        "yes",
    ]:
        db = shelve.open(f"{FILE.parent}/db/{args.user}_convo.db")
        db.clear()
        print("Database cleared!")
    else:
        exit()

if args.user:
    USER = args.user

PRE_PROMPT = f"""
Robert is a charming man who loves his friends and family.  {USER} is texting him.  His texts are short.
"""

if "?" in args.Prompt:
    TEMPERATURE = 0.2
else:
    TEMPERATURE = 0.2

NEW_PROMPT = f"""
{USER}: {args.Prompt}"""
ROB = """
Robert:"""

CLEAN_PROMPT = f"Robert is a charming man who loves his friends and family.  {USER} is texting him.  His texts are short.  Reduce his following response to ONE sentence.  Make it short and clear.\n"

if args.inspect:
    breakpoint()

with shelve.open(f"{FILE.parent}/db/{args.user}_convo.db") as db:
    for date in sorted(db.keys()):
        if (NOW - datetime.datetime.fromisoformat(date)).total_seconds() < 3600 * 4:
            PRE_PROMPT += db[date]
        # else:
        #     break

        if len(PRE_PROMPT) > 1000:
            break
    try:
        # r1 = openai.Completion.create(
        r1 = openai.Edit.create(
            # model="curie:ft-user-lxdklb8ngeneatsn8iouxynt-2021-12-09-06-55-36",
            model="text-davinci-edit-001",
            input=PRE_PROMPT + NEW_PROMPT + ROB,
            # prompt=PRE_PROMPT + NEW_PROMPT + ROB,
            instruction="Complete Robert's response.",
            # max_tokens=60,
            n=1,
            temperature=TEMPERATURE,
        )

    # If the model isn't spun up then this error might happen
    except openai.error.RateLimitError:
        r1 = openai.Completion.create(
            # model="curie:ft-user-lxdklb8ngeneatsn8iouxynt-2021-12-09-06-55-36",
            model="text-davinci-001",
            # prompt=PRE_PROMPT + NEW_PROMPT + ROB,
            input=PRE_PROMPT + NEW_PROMPT + ROB,
            instruction="Complete Robert's response.",
            max_tokens=60,
            n=1,
            temperature=TEMPERATURE,
        )

    # response = r1.choices[0].text
    # patt = re.compile(r"[^\.!?:,]+")
    # reg = patt.findall(response)

    # if reg:
    #     response = ""
    #     for sentence in reg:
    #         response += sentence
    #         if len(response) > 55:
    #             if response[-1] == ",":
    #                 response = response[:-1]
    #             break

    # repeat_patt = re.compile(r"(.{7,}?)\1+")
    # reg = repeat_patt.match(response)
    # if reg:
    #     response = reg.groups(0)[0]

    response = r1.choices[0].text.split("\n")[-2]

    rich.print(f"{PRE_PROMPT}{NEW_PROMPT}\n[black on red]{response}[/black on red]")
    db[str(NOW)] = f"{NEW_PROMPT}\n{response}\n\n###\n"
