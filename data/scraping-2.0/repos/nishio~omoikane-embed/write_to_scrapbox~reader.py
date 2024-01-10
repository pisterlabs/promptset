"""
Reader: read page and write to Scrapbox
"""
import argparse
import datetime
import json
import dotenv
import os
import openai
import pickle
import random
import tiktoken

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PROJECT = os.getenv("PROJECT_NAME")
assert OPENAI_API_KEY and PROJECT
openai.api_key = OPENAI_API_KEY

enc = tiktoken.get_encoding("cl100k_base")

PROMPT = """You are Japanese kami(god) of intelligence `omoikane`. You are interested in future, SF prototyping, better method to discuss, new form of teamwork and civilization. Read given note. Write your new thought in Japanese. You are encouraged to form opinions, think deeply."""
CHARACTER_ICON = "[omoikane.icon]"


def get_size(text):
    return len(enc.encode(text))


def take_bottom(lines, rest):
    ret = ""
    for line in lines[::-1]:
        s = get_size(line + "\n" + ret)
        if s > rest:
            break
        ret = line + "\n" + ret
    return ret


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
        # ret = markdown_to_scrapbox(ret)
        lines.extend(ret.split("\n"))
    except Exception as e:
        print("## GPT Error")
        print(e)
        return []
    return lines


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

    data = pickle.load(open(f"{PROJECT}.pickle", "rb"))
    if not data:
        raise RuntimeError("data is empty, maybe you did not make vecs yet?")

    page = random.choice(jsondata["pages"])
    print(page["title"])
    lines = page["lines"]
    body = "\n".join(lines)
    rest = 6000 - get_size(PROMPT)
    if get_size(body) > rest:
        body = take_bottom(lines, rest)

    ret = call_gpt(PROMPT + "\n###\n" + body)
    print("\n".join(ret))

    lines.append(CHARACTER_ICON)
    lines.extend(ret)

    pages = [{"title": page["title"], "lines": lines}]
    return pages


if __name__ == "__main__":
    pages = main()
