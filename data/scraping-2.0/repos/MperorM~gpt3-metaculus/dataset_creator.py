# Everything related to generating the dataset using the metaculus API.

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

#
# retrieve questions into giant dataset and save to disk
#

offset = 0
questions = pd.DataFrame()
for i in range(0, 5):
    url = f'https://www.metaculus.com/api2/questions/?limit=100&offset={offset}'
    response = requests.get(url)
    next_questions: dict = response.json()
    questions = questions.append(pd.DataFrame.from_dict(next_questions['results']))
    offset += 100


# save to disk
questions.to_pickle('metaculus_questions.pkl')


#
# get detailed descriptions for every binary question
#

questions = pd.read_pickle('metaculus_questions.pkl')

# filter away non-binary questions and questions without resolution
questions = questions.loc[questions['possibilities'] == {'type': 'binary'}]
questions = questions[questions['resolution'].notna()]
questions = questions.loc[questions['resolution'] != -1.0] # remove ambiguously resolved questions
questions = questions.loc[questions['resolution'] != None] # remove unresolved questions
questions = questions[questions['prediction_timeseries'].notna()] # remove questions without community predictions

# create get request for each url
detailed_questions = []
for question in questions.url:
    detailed_questions.append(requests.get(question).json())
questions = pd.DataFrame.from_dict(detailed_questions)

# get the average community prediction for first ten timeseries
community_preds = [question[0:20] for question in questions.prediction_timeseries]
means = []
for pred in community_preds:
    means.append(st.mean([d['community_prediction'] for d in pred]))
questions['community_mean'] = means

# save to disk
questions.to_pickle('metaculus_detailed_questions.pkl')