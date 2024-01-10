# Everything related to generating predictions using the GPT-3 model.

import numpy as np
import re
from sklearn.metrics import mean_squared_error
import pandas as pd
import statistics as st
import requests
import json
pd.set_option('display.max_columns', None)
import os
import openai
from gpt_prompt import short_gpt_prompt

# load questions
questions = pd.read_pickle('metaculus_detailed_questions.pkl')

# # to save space
# questions = questions[:10]
# questions = questions['resolution', 'community_mean', 'title'] # keep only necessary features

# Initialize GPT-3
openai.organization = os.getenv("OPENAI_ORG")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Append question to prompt
for index, row in questions.iterrows():
    questions.at[index, 'prompt'] = short_gpt_prompt + row['title'] + '\nLikelihood:'

# create completions
for index, row in questions.iterrows():
    # "text-davinci-002"    most expensive, most accurate
    # "text-ada-001"        cheapest, least accurate
    completion = openai.Completion.create(engine="text-davinci-002", prompt=row.prompt, max_tokens=2)
    completion = completion.choices[0].text

    questions.at[index, 'completions'] = completion

# save completions to disk
questions.to_pickle('gpt-davinci-completions.pkl')

questions = pd.read_pickle('gpt-davinci-completions.pkl')

# Sanitize GPT's completions
def gpt_cleanser(x):
    match = re.findall(r"[0-9]?[0-9]", x)
    if len(match) > 0:
        return match[0]
    else:
        return np.NaN

questions.completions = questions.completions.apply(lambda x: gpt_cleanser(x))
questions = questions[questions['completions'].notna()]
questions.completions = questions.completions.apply(lambda x: int(x) / 100)

# calculate brier score for community predictions
community_preds_MSE = mean_squared_error(questions.resolution, questions.community_mean)

# calculate brier score for GPT
gpt_MSE = mean_squared_error(questions.resolution, questions.completions)

# calculate brier score for random predictions
questions['random'] = np.random.random_sample(size=len(questions))
random_MSE = mean_squared_error(questions.resolution, questions.random)

# find percentage of questions that resolved negative. (66%)
len(questions.loc[questions.resolution == 0.0]) / len(questions)