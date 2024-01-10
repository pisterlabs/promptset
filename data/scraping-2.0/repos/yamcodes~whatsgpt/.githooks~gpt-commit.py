#!/usr/bin/env python

# this script uses https://github.com/markuswt/gpt-commit. Thanks Markus!

import asyncio
import openai
import os
import subprocess
import sys
from dotenv import load_dotenv

# Get the arguments

args = sys.argv[1:]

number_of_args = len(args)

# if number_of_args is more than 1, apologize and quit

if number_of_args > 1:
    print("I'm so, so sorry, but I can only handle one argument at a time.")
    print("This argument is the extra commit info you would like to pass ChatGPT.")
    print("Please try again, by either providing no arguments, or one argument.")
    print("If you wish to add more than one word, please use quotes.")

    quit()

DIFF_PROMPT = "Generate a succinct summary of the following code changes:"
COMMIT_MSG_PROMPT = 'Generate a descriptive commitmessage from these summaries.' \
    'Add a commit title and adhere to Angular commit conventions.' \
    'Make sure you include a tag that adheres to the conventions.' \
    'Do not  reference an issue.' \
    'After the title, use a bulletpoint list where each bullet point starts with "+ ", to describe each change.' \
    'Start each sentence with a noun, except of course for the tag.' \
    'Everything must be lowercase.' \
    'Your title must be limited to 50 characters, and this includes the tag.' \
    'Your lines must be limited to 72 characters.'
if number_of_args == 1:
    COMMIT_MSG_PROMPT += " also, " + str(args[0])
PROMPT_CUTOFF = 10000
openai.organization = os.getenv("OPENAI_ORG_ID")
load_dotenv()


def commit(message):
    # will ignore message if diff is empty
    return subprocess.run(["git", "commit", "--message", message,
                           "--edit"]).returncode


openai.api_key = os.environ["OPENAI_API_KEY"]


def get_diff():
    arguments = [
        "git", "--no-pager", "diff", "--staged", "--ignore-space-change",
        "--ignore-all-space", "--ignore-blank-lines"
    ]
    diff_process = subprocess.run(arguments, capture_output=True, text=True)
    diff_process.check_returncode()
    return diff_process.stdout.strip()


def parse_diff(diff):
    file_diffs = diff.split("\ndiff")
    file_diffs = [file_diffs[0]
                  ] + ["\ndiff" + file_diff for file_diff in file_diffs[1:]]
    chunked_file_diffs = []
    for file_diff in file_diffs:
        [head, *chunks] = file_diff.split("\n@@")
        chunks = ["\n@@" + chunk for chunk in reversed(chunks)]
        chunked_file_diffs.append((head, chunks))
    return chunked_file_diffs


def assemble_diffs(parsed_diffs, cutoff):
    # create multiple well-formatted diff strings, each being shorter than cutoff
    assembled_diffs = [""]

    def add_chunk(chunk):
        if len(assembled_diffs[-1]) + len(chunk) <= cutoff:
            assembled_diffs[-1] += "\n" + chunk
            return True
        else:
            assembled_diffs.append(chunk)
            return False

    for head, chunks in parsed_diffs:
        if not chunks:
            add_chunk(head)
        else:
            add_chunk(head + chunks.pop())
        while chunks:
            if not add_chunk(chunks.pop()):
                assembled_diffs[-1] = head + assembled_diffs[-1]
    return assembled_diffs


async def complete(prompt):
    completion_resp = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user",
            "content": prompt[:PROMPT_CUTOFF + 100]
        }],
        max_tokens=128)
    completion = completion_resp.choices[0].message.content.strip()
    return completion


async def summarize_diff(diff):
    assert diff
    return await complete(DIFF_PROMPT + "\n\n" + diff + "\n\n")


async def summarize_summaries(summaries):
    assert summaries
    return await complete(COMMIT_MSG_PROMPT + "\n\n" + summaries + "\n\n")


async def generate_commit_message(diff):
    if not diff:
        # no files staged or only whitespace diffs
        return "Fix whitespace"

    assembled_diffs = assemble_diffs(parse_diff(diff), PROMPT_CUTOFF)
    summaries = await asyncio.gather(
        *[summarize_diff(diff) for diff in assembled_diffs])
    return await summarize_summaries("\n".join(summaries))


async def main():
    try:
        diff = get_diff()
        commit_message = await generate_commit_message(diff)
    except UnicodeDecodeError:
        print("gpt-commit does not support binary files", file=sys.stderr)
        commit_message = "# gpt-commit does not support binary files. Please enter a commit message manually or unstage any binary files."

    if "--print-message" in sys.argv:
        print(commit_message)
    else:
        exit(commit(commit_message))


if __name__ == "__main__":
    asyncio.run(main())
