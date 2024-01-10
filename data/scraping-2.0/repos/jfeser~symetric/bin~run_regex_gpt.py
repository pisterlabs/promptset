#!/usr/bin/env python

__doc__ = """
Usage: run_regex_gpt.py [options]

Options:
    -s, --sketch <sketch>  Sketch to use
"""

from docopt import docopt
import openai
import os
import subprocess
import json


use_sketch = True
use_nl = False

cur_dir = os.path.dirname(os.path.realpath(__file__))
prompt_file = (
    cur_dir
    + "/../"
    + (
        "regex_prompt.txt"
        if use_sketch
        else ("regex_nosketch_nl_prompt.txt" if use_nl else "regex_nosketch_prompt.txt")
    )
)


def main(args):
    if use_sketch:
        sketch = args["--sketch"]

    openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.organization = os.environ["OPENAI_ORGANIZATION"]

    with open(prompt_file, "r") as f:
        prompt = f.read()

    if use_sketch:
        prompt += "\nSketch:\n"
        prompt += sketch + "\n"

    prompt += "\n"
    cmd = [cur_dir + "/../_build/default/bin/print_regex.exe"]
    if use_nl:
        cmd += ["-nl"]
    prompt += subprocess.check_output(
        cmd,
        encoding="utf-8",
    )
    prompt += "\n"
    prompt += "Program:\n"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
        ],
        n=10,
    )
    progs = [r["message"]["content"] for r in response.choices]
    with open("full_prompt", "w") as f:
        f.write(prompt)
    print(json.dumps(progs))
    # print(
    #     subprocess.check_output(
    #         ["_build/default/bin/print_bench.exe", "-scaling", "2"],
    #         input=prog,
    #     ).decode("utf-8")
    # )


if __name__ == "__main__":
    main(docopt(__doc__))
