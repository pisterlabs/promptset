"""Story Bot: OpenAI

https://platform.openai.com/docs/introduction
https://github.com/openai/openai-python
"""
import os

import openai
import backoff
from dotenv import load_dotenv


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
@backoff.on_exception(backoff.expo, openai.error.ServiceUnavailableError)
def generate_story(user_prompt: str):
    system_prompt = "You are a masterful storyteller. Please respond with a " \
                    "short story based on the following prompt."
    raw_output = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=None,
        temperature=0.5,
    )
    reply = raw_output.choices[0].message.content.strip()
    return reply


if __name__ == '__main__':
    from time import perf_counter
    from datetime import timedelta

    start = perf_counter()
    print(generate_story("Tell me a fantasy story about the adventures of a hamster"))
    stop = perf_counter()
    print(f"\nTotal Time: {timedelta(seconds=stop - start)}")
