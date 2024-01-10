import os
import openai
from icecream import ic
from pathlib import Path
import random
import datetime
import math
import pandas as pd
from io import StringIO
import re
import time
from gptchat_data import generate_chat_text_dir

openai.api_key_path = Path.home() / ".openai-api-key"


"""
Topics that are deliberately benign or even asinine.
"""
TOPICS = [
    "the weather",
    "favourite pets and why",
    "nostalgia for childrens TV programmes",
]


def get_prompt(topic: str, number_of_msg: int, number_of_users: int) -> str:
    prompt = (
        f"Create the conversation between {number_of_users} users of a chat forum. The topic is {topic}. There should be {number_of_msg} distinct messages."
        """
    The results must a CSV file with the columns 'msg_id', 'username', 'text'. The 'text' column must be enclosed with double quotes.
    """
    )
    return prompt


def get_response(prompt):
    response = None
    attempts = 0

    # Retry upto 50 times if there an error from the OpenAI API
    # There are better ways to implement this. See:
    # https://platform.openai.com/docs/guides/rate-limits/retrying-with-exponential-backoff
    while response is None:
        attempts += 1
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=0.7,
                max_tokens=3500,
                top_p=1,
                frequency_penalty=0.5,
                presence_penalty=0.5,
            )
        except (
            openai.error.ServiceUnavailableError,
            openai.error.RateLimitError,
        ) as openai_error:
            if attempts > 50:
                raise openai_error
            time.sleep(30)

    return response


def get_sampling_threads(
    num_unique_threads: int, max_msg_per_thread: int, mode_msg_per_thread: int
):

    now_str = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    result_df = pd.DataFrame()

    for thread_id in range(num_unique_threads):

        the_topic = random.choice(TOPICS)
        number_of_msg = int(
            random.triangular(1, max_msg_per_thread, mode_msg_per_thread)
        )

        # A bit of logic about a plausible number pf users vs length of thread
        min_users = 2 if number_of_msg > 2 else 1
        max_users = max(min_users, number_of_msg)
        number_of_users = (
            random.randrange(min_users, max_users)
            if max_users > min_users
            else min_users
        )

        ic(thread_id, the_topic, number_of_msg, number_of_users)

        the_prompt = get_prompt(the_topic, number_of_msg, number_of_users)
        # ic(the_prompt)

        next_df = None

        response = get_response(the_prompt)
        ic(response["choices"][0]["text"])

        next_df = pd.DataFrame(
            {
                "thread_id": thread_id,
                "number_of_msg": number_of_msg,
                "number_of_users": number_of_users,
                "raw_text": response["choices"][0]["text"],
            },
            index=[thread_id],
        )

        if next_df is None:
            raise ValueError("Unable to get parable response from OpenAI")

        next_df.to_csv(generate_chat_text_dir / f"individual_thread_{thread_id}.csv")

        result_df = pd.concat([result_df, next_df])
        # Save and clobber on every loop, in case of interruption
        result_df.to_json(
            generate_chat_text_dir / f"combined_raw_threads_{now_str}.json"
        )
        # ic(response)

    return result_df


def combine_json_files():

    in_files = generate_chat_text_dir.glob("combined_raw_threads_*.json")

    df = pd.DataFrame()

    for in_file in in_files:
        in_df = pd.read_json(in_file)

        df = pd.concat([df, in_df], ignore_index=True)
        df["thread_id"] = df.index
        df = df.head(300)
        df.to_json(generate_chat_text_dir / f"openai_thread_pool.json")
        ic(len(in_df), len(df))


if __name__ == "__main__":

    # Create a number of example thread using then ChatGPT API
    result_df = get_sampling_threads(
        num_unique_threads=300, max_msg_per_thread=40, mode_msg_per_thread=5
    )

    # Only required if needed to combine the output from multiple runs:
    # combine_json_files()
