#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yudai Tanaka

Solve the translated non-comprehension questions of the NMLE in Japan by ChatGPT API.

Setup requires:
- `YOUR_OPENAI_API_KEY`: Your OpenAI API key
- `YOUR_QUESTION_CSV_FILE_NAME`: Name of your question CSV file
- `PATH_TO_QUESTION_FOLDER`: Path to your question CSV file's folder
- `PATH_TO_RESULT_FOLDER`: Path to save the result
"""

# Install the packages
import openai
import pandas as pd

openai.api_key = "YOUR_OPENAI_API_KEY"


# Translation the original Japanese sentences into English ones and answer the translated questions
# For non-comprehension questions
def english_ask(name):
    df = pd.read_csv("PATH_TO_QUESTION_FOLDER" + name + ".csv", header=0, index_col=0)
    df["english"] = ""
    df["output"] = ""
    df["check"] = ""
    df["error"] = ""
    for i in range(len(df)):
        english = []
        english = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Translate the following Japanese into English.",
                },
                {"role": "user", "content": str(df.iloc[i, 3]) + str(df.iloc[i, 4])},
            ],
            temperature=0,
        )
        df.iloc[i, 7] = english["choices"][0]["message"]["content"]

        res = []
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Answer the following questions with reasons.",
                },
                {"role": "user", "content": df.iloc[i, 7]},
            ],
            temperature=0,
        )
        df.iloc[i, 8] = res["choices"][0]["message"]["content"]

    df.to_csv("PATH_TO_RESULT_FOLDER" + name + ".csv")


english_ask("YOUR_QUESTION_CSV_FILE_NAME")
