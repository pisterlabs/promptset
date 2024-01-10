# coding:utf-8

import argparse
import os
import random
import sys

import arxiv
import openai
from slack_sdk import WebClient as SlackClient
from slack_sdk.errors import SlackApiError

from ..utils import toRED


def get_arxiv_summary(result: arxiv.Result) -> str:
    system = """与えられた論文の要点を3点のみでまとめ、以下のフォーマットで日本語で出力してください。

```
タイトルの日本語訳
・要点1
・要点2
・要点3
```
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"title: {result.title}\nbody: {result.summary}"},
        ],
        temperature=0.25,
    )
    if isinstance(response, dict):
        summary = response["choices"][0]["message"]["content"]
        title, *body = summary.split("\n")
        body = "\n".join(body)
        return f"""発行日: {result.published.strftime("%Y-%m-%d %H:%M:%S")}
URL: {result.entry_id}
Title: "{result.title}"
タイトル: 「{title}」
-------------------------
{body}
"""
    else:
        return "Error"


def main(argv: list = sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Summarize arxiv papers with ChatGPT and post it to Slack.",
        add_help=True,
    )
    parser.add_argument(
        "-openai",
        "--OPENAI-API-KEY",
        type=str,
        default=os.getenv("OPENAI_API_KEY", ""),
        help="The openai's api key.",
    )
    parser.add_argument(
        "-slack",
        "--SLACK-API-TOKEN",
        type=str,
        default=os.getenv("SLACK_API_TOKEN", ""),
        help="The slack api token.",
    )
    parser.add_argument(
        "-channel",
        "--SLACK-CHANNEL",
        type=str,
        default=os.getenv("SLACK_CHANNEL", ""),
        help="Which channel to post the arxiv summary.",
    )
    parser.add_argument(
        "-Q",
        "--query",
        type=str,
        default=os.getenv("ARXIV_QUERY", "abs:GPT AND cat:cs.AI"),  # Default: Abstract に "GPT" という文字を含む、AI関連の論文。
        help="The search query of Arxiv. (See 'https://info.arxiv.org/help/api/user-manual.html#query_details'.)",
    )
    parser.add_argument("-N", "--num", type=int, default=3, help="How many papers to post.")
    args = parser.parse_args(argv)

    # Set openai API key.
    openai.api_key = args.OPENAI_API_KEY
    # Initialize the Slack Client.
    client = SlackClient(token=args.SLACK_API_TOKEN)
    # Search arxiv paper.
    search = arxiv.Search(
        query=args.query,
        max_results=20,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    for i, result in enumerate(sorted(search.results(), key=lambda k: random.random()), start=1):
        try:
            message = f"今日の論文です！ {i}本目\n" + get_arxiv_summary(result)
            parent_message = client.chat_postMessage(channel=args.SLACK_CHANNEL, text=message)
            child_message = client.chat_postMessage(
                channel=args.SLACK_CHANNEL, text=result.summary, thread_ts=parent_message["ts"]
            )

        except SlackApiError as e:
            print(f"Error posting message: {toRED(str(e))}")

        if i >= args.num:
            break
