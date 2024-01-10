
import dotenv
import openai
import time
import os
import json
import pickle
import datetime
import random
import tiktoken
import re
import requests
import argparse
from urllib.parse import quote
from utils import markdown_to_scrapbox

CHECK_ROBOT_IN_UPDATES = True  # if True, check if robot is in the latest updates

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PROJECT = os.getenv("PROJECT_NAME")
assert OPENAI_API_KEY and PROJECT
openai.api_key = OPENAI_API_KEY

PROMPT = """
You will be shown Conversation among people. Please read, memorize, and understand Task Conversation, then complete the task under the guidance of Task Introduction.
### Conversation
{conversation}
### Task
1 - Conclude all possible topics in the conversation with concise spans.
2 - Determine the chat range of each topic. These ranges should be a set of non-intersecting. Ignore small range less than 3 lines.
3 - Conclude a summary of each chat with brief sentences in Japanese.
4 - Report topic in Japanese, summary and range resutls in JSON format only with the assigned keys: 'topic', 'summary', 'startline', 'endline', 'index'. Startline is the content of the beginning line. Index is integer line number. Output shuold be a list of JSON objects `[...]`.
"""

LESS_INTERSTING = "___BELOW_IS_LESS_INTERESTING___"

enc = tiktoken.get_encoding("cl100k_base")


def get_size(text):
    return len(enc.encode(text))


def read_note_from_scrapbox(url):
    """
    url example: https://scrapbox.io/nishio/%F0%9F%A4%962023-08-13_07:08
    """

    api_url = re.sub(
        r"(https://scrapbox\.io)/([^/]+)/([^/]+)", r"\1/api/pages/\2/\3", url
    )
    page = requests.get(api_url).json()  # currently not supported private project
    return page["title"], [line["text"] for line in page["lines"]]


def call_gpt(prompt, model="gpt-4"):
    print("# Call GPT")
    print("## Prompt")
    print(prompt)
    if args.skip_gpt:
        print("## Skipped")
        return ["GPT Skipped"]
    print("--- End of Prompt")

    messages = [{"role": "system", "content": prompt}]
    lines = []
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.0,
            # max_tokens=max_tokens,
            n=1,
            stop=None,
        )
        ret = response.choices[0].message.content.strip()
        print("## GPT Response")
        print(ret)
        print("--- End of GPT Response")
        ret = markdown_to_scrapbox(ret)
        lines.extend(ret.split("\n"))
    except Exception as e:
        print("## GPT Error")
        print(e)
        return []
    return lines


def take_bottom(lines, rest):
    ret = ""
    for line in lines[::-1]:
        s = get_size(line + "\n" + ret)
        if s > rest:
            break
        ret = line + "\n" + ret
    return ret


def main():
    global args
    parser = argparse.ArgumentParser(description="Process a URL")
    parser.add_argument("--url", type=str, help="The URL to process", required=False)
    parser.add_argument(
        "--skip-gpt",
        action="store_true",
        help="skip GPT API call for tests",
    )

    args = parser.parse_args()

    jsondata = json.load(open(f"{PROJECT}.json"))
    buf = []
    for page in jsondata["pages"]:
        body = "\n".join(page["lines"])
        size = get_size(body)
        buf.append((size, page["title"], body))

    buf.sort(reverse=True)
    longest_page_size = buf[0][0]
    longest_page_title = buf[0][1]
    body = buf[0][2].split("\n")
    bottom = take_bottom(body, 4000)
    p = PROMPT.format(conversation=bottom)

    message = [
        "🤖",
        "このページは毎日自動生成されて上書きされます。",
        f"最も長いページ: {longest_page_title} ({longest_page_size} tokens)", ""]
    ret = call_gpt(p)
    print("\n".join(ret))
    try:
        items = json.loads("\n".join(ret))
        for item in sorted(items, key=lambda x: x["index"]):
            message.append(item["topic"])
            message.append(item["summary"] + "[neko.icon]")
            message.append("> " + item["startline"])
            message.append("> ...")
            message.append("> " + item["endline"])
            message.append("")
    except Exception as e:
        print(e)

    print("# Message")
    print("\n".join(message))

    # title, lines = read_note_from_scrapbox("https://scrapbox.io/unnamed-project/%E9%9B%91%E8%AB%87%E3%83%9A%E3%83%BC%E3%82%B89")
    return [{"title": "🤖", "lines": message}]




if __name__ == "__main__":
    pages = main()
    for page in pages:
        print(page["title"])
        print("\n".join(page["lines"]))
        print()
