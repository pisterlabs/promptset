#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yudai Tanaka

Solve the original Japanese text comprehension questions of the NMLE in Japan by ChatGPT API

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

# Answer the Original Japanese questions
# For comprehension questions
df = pd.read_csv(
    "PATH_TO_QUESTION_FOLDER" + "YOUR_QUESTION_CSV_FILE_NAME" + ".csv",
    header=0,
    index_col=0,
)
df["output"] = ""
df["check"] = ""
df["error"] = ""
for i in range(len(df) // 2):
    res = []
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "次の質問に理由を添えて答えよ"},
            {
                "role": "user",
                "content": "Q1:"
                + df.iloc[i * 2, 3]
                + df.iloc[i * 2, 4]
                + "Q2:"
                + df.iloc[i * 2 + 1, 3]
                + df.iloc[i * 2 + 1, 4],
            },
        ],
        temperature=0,
    )
    df.iloc[i * 2, 8] = res["choices"][0]["message"]["content"]

df.to_csv("PATH_TO_RESULT_FOLDER" + "YOUR_QUESTION_CSV_FILE_NAME" + ".csv")
