# import redis
import os

import openai

# redis_client = redis.Redis(host='localhost', port=6379, db=0)
from report_cruncher.constants import OPENAI_API_KEY_kEY


def execute(prompt):
    # Call OpenAI API
    openai.api_key = os.getenv(OPENAI_API_KEY_kEY)
    # response = openai.Completion.create(
    #     engine="davinci",
    #     prompt=prompt,
    #     temperature=0.7,
    #     max_tokens=256,
    #     n=1,
    #     stop=None,
    # )
    # Extract the embedding from the response
    # embedding = response.choices[0].embedding

    # Store the embedding in Redis
    # redis_client.set("embedding", embedding)

    # return embedding
    response = openai.Completion.create(
        model="curie:ft-tpisoftware-2023-03-02-16-10-55",
        prompt=f"Summarize the following text:{prompt[0:2000]}",
        max_tokens=256,
        temperature=0
    )
    # print(response)
    text = response.choices[0].text
    # Store the text in Redis
    # redis_client.set("text", text)

    return text


def execute_question(article, question):
    # Call OpenAI API
    openai.api_key = os.getenv(OPENAI_API_KEY_kEY)

    response = openai.Completion.create(
        model="curie:ft-tpisoftware-2023-03-02-16-10-55",
        prompt=f"out of this context {article[0:2000]}, answer this question {question}",
        max_tokens=64,
        temperature=0
    )
    # print(response)
    text = response.choices[0].text.strip(" \n")
    # Store the text in Redis
    # redis_client.set("text", text)

    return text
