#!/usr/bin/env python3

from openai import OpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI key missing")

client = OpenAI(api_key=api_key)


def getChatResponse(chatHistory):
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=chatHistory, max_tokens=200)

        completionText = completion.choices[0].message.content
        return {"role": "assistant", "content": completionText}
    except Exception as e:
        raise e
