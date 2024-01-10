#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yudai Tanaka

Solve the translated comprehension questions of the NMLE in Japan by ChatGPT API with the optimized prompts.

Setup requires:
- `YOUR_OPENAI_API_KEY`: Your OpenAI API key
- `YOUR_QUESTION_CSV_FILE_NAME`: Name of your question CSV file
- `PATH_TO_QUESTION_FOLDER`: Path to your question CSV file's folder
- `PATH_TO_RESULT_FOLDER`: Path to save the result
- `PATH_TO_PROMPT_CSV_FILE`: Path to the CSV file containing optimized prompts
"""

# Install the packages
import openai
import pandas as pd

openai.api_key = "YOUR_OPENAI_API_KEY"

# Load the optimized prompts
df_p = pd.read_csv("PATH_TO_PROMPT_CSV_FILE", header=0, index_col=0)
df_p.head()

tra = [
    df_p.iloc[0, 0],
    df_p.iloc[1, 0],
    df_p.iloc[0, 0],
    df_p.iloc[0, 0],
    df_p.iloc[1, 0],
    df_p.iloc[1, 0],
]
ex = [
    df_p.iloc[0, 1],
    df_p.iloc[1, 1],
    df_p.iloc[0, 1],
    df_p.iloc[0, 1],
    df_p.iloc[1, 1],
    df_p.iloc[1, 1],
]

# Translation the original Japanese sentences into plain English and answer the translated questions
# For comprehension questions
df = pd.read_csv(
    "PATH_TO_QUESTION_FOLDER" + "YOUR_QUESTION_CSV_FILE_NAME" + ".csv",
    header=0,
    index_col=0,
)
df["english"] = ""
df["output"] = ""
df["check"] = ""
df["error"] = ""
for i in range(len(df) // 2):
    english = []
    english = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": df_p.iloc[2, 0]},
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
    df.iloc[i * 2, 7] = english["choices"][0]["message"]["content"]

    res = []
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": df_p.iloc[2, 1]},
            {"role": "user", "content": df.iloc[i * 2, 7]},
        ],
        temperature=0,
    )
    df.iloc[i * 2, 8] = res["choices"][0]["message"]["content"]

df.to_csv("PATH_TO_RESULT_FOLDER" + "YOUR_QUESTION_CSV_FILE_NAME" + ".csv")
