import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


def sentiment_prompt(answer):
    return f"""Classify the sentiment in these tweets:\n\n1. \"Thank You\"\n2. \"You are not answering my question\"\n3. \"This was helpful\"\n4. \"okay\"\n5. \"This was rude\"\n6. \{answer}  \n\nTweet sentiment ratings:""",


def sentiment(answer):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=sentiment_prompt(answer),
        temperature=0,
        max_tokens=60,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
