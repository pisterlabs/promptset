import os
import ast
import json
import time
import openai
import threading

import pandas as pd
import numpy as np

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv(), override=True)  # read local .env file


client = openai.OpenAI(api_key=openai.api_key)

df = pd.read_csv("output.csv")
df = df[df["comments"].apply(lambda x: len(x) > 3)]

os.mkdir("issues_txt")

indexes = []

for index, row in df.iterrows():
    writer = row["writer"]
    title = row["title"]
    is_open = row["is_open"]
    issue_body = row["body"]
    comments = row["comments"]
    comments = ast.literal_eval(comments)

    issue = {
        "title": title,
        "author": writer,
        "is_open": is_open,
        "issue": issue_body,
        "comments": comments,
    }

    indexes.append(index)

    file_path = f"issues_txt/{index}.txt"
    json_string = json.dumps(issue, indent=4)

    with open(file_path, "w") as f:
        f.write(json_string)


folder_path = "issues_txt"

PROMPT = """
You are a Github Issue data processor.
It is github issue and its comments from a Github Issue.

You have data in this format:
{
    title: Title of the Issue,
    author: author of the issue,
    is_open: is the issue still open or closed,
    issue: content of the issue,
    comments: [
        [
            author of the comment,
            comment content
        ],
        [
            author of the comment,
            comment content
        ],
        [
            author of the comment,
            comment content
        ],
        [
            author of the comment,
            comment content
        ]
        ...
    ]
}

First is the title of the issue.
Second is the author of the issue.
Third is the issue still open or closed.
Fourth is the content of the issue.
Others are answers to the issue.

All messages are from the same issue in github. Read all the messages, infer the answer and create question(issue) and answer pair by considering conditions below:

Question need to be complete and include thread name.
If question or answer includes code blocks, include code blocks as well.
Include title and issue content and full issue in question field completely with also code blocks if exists.
Please do not summarize the answer. Evaulate and explain the answer as detailed as possible with all necessary information which is also has code blocks.

Infer, evaulate and create Full Issue: Detailed Answer Pair
Return a valid JSON as the final result, if there is no answer in the messages, return null. Thank you is not an answer, this data will be used for training so please remove unnecessary data.
Give me a JSON file with the following format in markdown format:
```json
{
"question": "The question",
"answer": "The answer" or None
}
```
"""


def process_txt(number_of_txt):
    file_path = f"{folder_path}/{number_of_txt}.txt"

    if not os.path.exists(file_path):
        return

    with open(file_path, "r") as file:
        contents = file.read()

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        response_format={"type": "json_object"},
        temperature=0.9,
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": str(contents)},
        ],
    )

    result = response.choices[0].message.content
    print(result)

    return result


def run_with_timeout(func, args, timeout):
    result = [None]  # A mutable object to store the result of func

    def target(result_list):
        result_list[0] = func(*args)

    thread = threading.Thread(target=target, args=(result,))
    thread.start()
    thread.join(timeout)  # Wait for the specified timeout
    if thread.is_alive():
        print("Function timed out")
        return None  # Function did not complete within timeout
    else:
        return result[0]  # Function completed within timeout


os.mkdir("issues_json")

for index in indexes:
    full_question = {"full_question": f"{title}\n{issue}"}

    print(f"Processing {index}...")
    result = run_with_timeout(process_txt, [index], 90)

    if result is None:
        continue

    if result is not None:
        txt_file_path = f"issues_txt/{index}.txt"
        with open(txt_file_path, "r") as file:
            contents = file.read()

        contents = json.loads(contents)
        title = contents["title"]
        issue = contents["issue"]

        file_path = f"issues_json/processed{index}.json"

        result = json.loads(result)
        result.update(full_question)

        print(f"Saving {index}...")
        with open(file_path, "w") as file:
            file.write(json.dumps(result, indent=4))

        time.sleep(10)
