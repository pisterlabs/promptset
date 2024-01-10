import os
import pandas as pd
import random
import time
from ratelimiter import RateLimiter

import openai
from openai.error import APIError, RateLimitError

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_KEY_1")

def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (RateLimitError,),
):

    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)

            except errors as e:
                num_retries += 1

                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                delay *= exponential_base * (1 + jitter * random.random())

                time.sleep(delay)

            except Exception as e:
                raise e

    return wrapper

@retry_with_exponential_backoff
def send_completion_request(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=20,
    )
    return response

jokes = pd.read_csv("../other/jokes/cleaned_compiled_data/final_cleaned_jokes.csv")

if "category" not in jokes.columns:
    jokes["category"] = ""

joke_list = jokes["joke"].tolist()[:16001]

for index, joke in enumerate(joke_list):
    print(index)
    if not pd.isna(jokes.loc[jokes["joke"] == joke, "category"].iloc[0]):
        continue

    prompt = "Give the following joke a suitable category: " + joke
    try:
        response = send_completion_request(prompt)
        category = response["choices"][0]["text"].strip()
        jokes.loc[jokes["joke"] == joke, "category"] = category

        if (index + 1) % 10 == 0:
            jokes.to_csv("../other/jokes/cleaned_compiled_data/final_cleaned_jokes.csv", index=False)

    except Exception as e:
        jokes.to_csv("../other/jokes/cleaned_compiled_data/final_cleaned_jokes.csv", index=False)
        print("Exception: ", e)
        break

jokes.to_csv("../other/jokes/cleaned_compiled_data/final_cleaned_jokes.csv", index=False)
print("done")
