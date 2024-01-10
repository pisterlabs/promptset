import os

import openai

from aifail import retry_if, openai_should_retry

client = openai.OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    max_retries=0,  # disables openai's internal retry logic
)


@retry_if(openai_should_retry)
def gpt4(messages):
    response = client.chat.completions.create(model="gpt-4", messages=messages)
    print(response.choices[0].message.content)


gpt4(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant designed to output JSON.",
        },
        {
            "role": "user",
            "content": "Who won the world series in 2020?",
        },
    ]
)
