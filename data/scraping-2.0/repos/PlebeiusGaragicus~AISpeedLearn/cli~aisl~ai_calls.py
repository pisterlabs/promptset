import os
from time import sleep
import logging
logger = logging.getLogger(__name__)

import openai

def setup_magic():
    openai.api_key = os.getenv("OPENAI_API_KEY")



    # response = openai.ChatCompletion.create(
    #     model="text-davinci-003",
    #     prompt=prompt,
    #     temperature=0.9,
    #     max_tokens=2048, #??
    #     top_p=1,
    #     frequency_penalty=0.0,
    #     presence_penalty=0.6,
    #     # stop=[" Human:", " AI:"]
    # )
def run_prompt(prompt: str) -> str:
    def api_call():
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=2048,
        )

        # print(response)
        return response.choices[0].message.content

    attempts = 0
    sleep_time = 10
    while attempts < 3:
        try:
            return api_call()
        except openai.error.RateLimitError as e:
            logger.fatal("OpenAI API call failed: RateLimitError")
            logger.exception(e)
            attempts += 1
            logger.info(f"Sleeping for {sleep_time} seconds")
            sleep(sleep_time)
            sleep_time *= 1.8
            continue
        except openai.error.APIError as e:
            logger.exception(e)

    return "OpenAI API call failed 3 times. Please try again later."
