import os
import re

from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

regex = r".+sentiment:\s*(-?\d(?:\.\d+))$"

MODEL = "gpt-3.5-turbo-1106"


def generate_readable(content):
    summary = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "Current date: December 8th, 2023.\n\nYou will read a part of a text that is sourced from a website",
            },
            {"role": "user", "content": content},
            {
                "role": "system",
                "content": "Filter out irrelevant information such as ads or reading suggestions. Include timestamps, up/downvotes, and relevant tooltips and notices. Format the post/conversation. Format should be <username> (<absolute date. convert relative dates to absolute yyyy-mm-dd>): <content>. Replies should be indented with prefix '-' to indicate that it is a reply. Do NOT alter the original text",
            },
        ],
    )

    return summary.choices[0].message.content


def check_relevance(content, topic):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You will read a part of a text that is sourced from a website. Understand the topic of the text",
            },
            {"role": "user", "content": content},
            {
                "role": "system",
                "content": f"Is the text relevant to '{topic}'? (yes/no)",
            },
        ],
    )

    c = response.choices[0].message.content.lower()

    return "yes" in c


def sentimental_analysis(content, about):
    match = None

    for i in range(2):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You will read a part of a text that is sourced from a website. Understand the sentiment of the text",
                },
                {"role": "user", "content": content},
                {
                    "role": "system",
                    "content": f"Give a brief sentiment about the post/conversation about {about}. End your response with 'Sentiment: <number>', where the number ranges from -1.0 to 1.0, -1.0 being strongly disagree and 1.0 being strongly agree.",
                },
            ],
        )

        match = re.search(
            regex, response.choices[0].message.content, re.IGNORECASE | re.DOTALL
        )

        if match:
            break

        print(f"ChatGPT failed to give a parseable response. Trying again... ({i + 1})")
    else:
        return None

    review, rating = match.group(0), match.group(1)

    return [review, float(rating)]
