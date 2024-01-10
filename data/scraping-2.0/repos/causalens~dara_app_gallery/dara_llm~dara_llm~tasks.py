"""
Copyright 2023 Impulse Innovations Limited


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os

import openai
import pandas as pd
from statsmodels.stats.stattools import jarque_bera

from dara_llm.definitions import COEFFICIENTS, MODEL

openai.api_key = os.getenv('OPENAI_API_KEY')


def query_chat_gpt(question: str) -> str: 
    """
    Queries ChatGPT (3.5-Turbo) with a question that includes the context of the problem and model.

    :param question: The given query string.
    """
    if question == '':
        return ''

    # Provide the overall context of the problem.
    prompt = (
        'I have a dataset with the following features: '
        'TV Marketing Spend, Radio Marketing Spend, and Newspaper Marketing Spend.'
        'These features are used to predict the target which is the number of Sales.'
        'To predict Sales, I built an Ordinary Least Squares Model.'
    )

    # Provide coefficients as additional context.
    prompt += 'The OLS model has the following coefficients and corresponding p values: '
    for _, row in COEFFICIENTS[COEFFICIENTS['Feature'] != 'const'].iterrows():
        prompt += f"{row['Feature']} has the coefficient {row['Coefficient']} with a p-value of {row['P-Values']}. "
    for _, row in COEFFICIENTS[COEFFICIENTS['Feature'] == 'const'].iterrows():
        prompt += f"The intercept has a value of {row['Coefficient']} with a p-value of {row['P-Values']}."

    # Provide performance metrics as additional context.
    prompt += (
        f'The model has an F-Statistic of {round(MODEL.fvalue, 2)}, '
        f'a R-Squared of {round(MODEL.rsquared, 2)}, and a Log Likelihood of {round(MODEL.llf, 2)}'
    )

    # Provide information on the residual distribution as additional context.
    _, _, skew, kurtosis = jarque_bera(MODEL.wresid)
    prompt += (
        f"The skewness of the model's residual distribution is {round(skew, 2)}. "
        f"The kurtosis of the model's residual distribution is {round(kurtosis, 2)}. "
    )

    prompt += question
    prompt += 'Answer in three sentences.'

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        temperature=1,
        messages=[{'role': 'user', 'content': prompt}],
    )
    return response.choices[0].message['content']
