import csv
import os

import numpy.random
import pandas as pd
import openai
from tqdm import tqdm

from stance_detection.stance_prompts import *

FILE_NAME = "gov_all.csv"
file = open(FILE_NAME, mode='a', newline='')


def get_data():
    believable_by_few = pd.read_csv('../believable_by_few.csv', sep=',')
    believable_by_many = pd.read_csv('../believable_by_many.csv', sep=',')
    return believable_by_few["text"], believable_by_many["text"]


def write_to_csv(tweet, stance, response, filename=FILE_NAME):
    # Check if the file exists already
    file_exists = os.path.isfile(filename)
    # Create a CSV writer object
    writer = csv.writer(file)
    # If the file doesn't exist yet, write the header row
    if not file_exists:
        header = ['tweet', 'result', 'response']
        writer.writerow(header)
    # Write the data row
    data_row = [tweet, stance, response]
    writer.writerow(data_row)


def detect(stance, text, prompts):
    user_prompt = f'A user tweeted the following message:"{text}".\n\n Does this tweet express {stance}?'
    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompts["system_msg"]},
                    {"role": "user", "content": prompts["q1_user"]},
                    {"role": "assistant", "content": prompts["q1_assistant"]},
                    {"role": "user", "content": prompts["q2_user"]},
                    {"role": "assistant", "content": prompts["q2_assistant"]},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                top_p=0
            )
            res = response.choices[0]['message']['content'].strip().replace('.', '').split(" ")[-1]
            # write_to_csv(text, res, response.choices[0]['message']['content'])
            if res == "Yes":
                return res, 1, response.choices[0]['message']['content']
            else:
                return res, 0, response.choices[0]['message']['content']
        except Exception as e:
            continue


if __name__ == "__main__":
    openai.api_key = open('../models/.openai.key.Lin').read().replace('\n', '').replace('\r', '').strip()
    few, many = get_data()
    numpy.random.seed(2)
    few_10 = few.sample(frac=1)[:10]
    many_10 = many.sample(frac=1)[:10]

    print(len(few_10))
    print(len(many_10))

    for text in tqdm(few_10):
        detect(narratives["gov"], text, gov)
    for text in tqdm(many_10):
        detect(narratives["gov"], text, gov)
