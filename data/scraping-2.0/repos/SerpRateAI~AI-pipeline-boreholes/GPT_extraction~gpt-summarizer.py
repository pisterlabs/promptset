"""
A file for making the summarization text from chatgpt
"""

import pandas as pd
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import openai



# we need to wait sometimes because of the API load limits, Tenacity lets us do that
# https://tenacity.readthedocs.io/en/latest/api.html#tenacity.wait.wait_exponential
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def summarize(text, debug=False):
    """
    This function uses a chatgpt 3.5 turbo model to summarize code into ten keywords
    """
    
    prompt = f"Please summarize the following text into ten keywords and explain why you picked each key word. The text to summarize is:{text}"
    
    if debug == True:
        print(prompt)
        
    return openai.ChatCompletion.create(
             model='gpt-3.5-turbo'
            ,messages=[{'role':'user', 'content':prompt},]
        )

if __name__=='__main__':
    
    # import the secret api key
    with open('api-key.txt', 'r') as f:
        openai.api_key = f.readline()
        
    df = pd.read_excel('Dataset_BA1B.xlsx')
    # the remarks are split across multiple columns in the excel spreadsheet
    # they need to be combined first
    df[['REMARKS1', 'REMARKS2', 'REMARKS4', 'REMARKS5',]] = df[['REMARKS1', 'REMARKS2', 'REMARKS4', 'REMARKS5',]].fillna('')
    df[['REMARKS1', 'REMARKS2', 'REMARKS4', 'REMARKS5',]] = df[['REMARKS1', 'REMARKS2', 'REMARKS4', 'REMARKS5',]].replace('none', '')
    df['REMARKS_ALL'] = (df['REMARKS1'] + ' ' + df['REMARKS2'] + ' ' + df['REMARKS4'] + ' ' + df['REMARKS5'])
    
    # Here is where the magic happens
    # each remark
    text_summary = []

    for n, row in enumerate(df.iterrows()):
        print('analysing row {n}'.format(n=n))
        remarks = row[1]['REMARKS_ALL']
        summed = summarize(remarks)
        text_summary.append(summed)
        
    
    summarized = pd.DataFrame(text_summary, index=df.index)
    summarized.to_csv('summarized.csv', index=True)
