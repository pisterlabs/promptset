from tqdm import tqdm
import json
import openai
import logging
import argparse
from utils import load_json, image_path_to_id, imageid_to_path, qdict_to_df, save_json
from collections import defaultdict
import os
import openai
from config import *
import openai
import random
import pandas as pd
from collections import Counter
import time


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
random.seed(1)
openai.api_key = ""


def combine_all_dfs(path):
    files = os.listdir(path)
    files.sort()
    print(files)
    ddf = pd.DataFrame()
    for file in files:
        if ".csv" in file:
            cur_df = pd.read_csv(path + file)
            ddf = pd.concat([ddf, cur_df])
    ddf.reset_index(drop=True, inplace=True)
    ddf.to_csv(path+"gpt3_predictions_train2014.csv")


def main():
    split = 'train2014'

    annotations = load_json(f'{data_root}/ok-vqa/mscoco_{split}_annotations.json')
    questions = load_json(f'{data_root}/ok-vqa/OpenEnded_mscoco_{split}_questions.json')
    captions = load_json(f'{data_root}/vqa/expansion/captions/captions_{split}_vqa.json')

    # Get most repeated answer
    ans_list = annotations['annotations']
    answer_map = {}
    for d in ans_list:
        q = str(d['question_id'])
        answers = [ans_dict['answer'] for ans_dict in d['answers']]
        most_common_words = [word for word, word_count in Counter(answers).most_common(1)]
        best_answer = most_common_words[0]
        answer_map[q] = best_answer

    df = qdict_to_df(questions)
    df["answer"] = df["question_id"].map(answer_map)
    df["caption"] = df["image_path"].map(captions)
    print("The head of the dataframe", df.head())
    print("The number of examples", df.shape)
    # Had to do in batches of 100
    STARTS = list(range(0, 5000, 100))
    for START in STARTS:
        END = START + 100
        print(START, END)
        train_examples = df[:8]
        test_examples = df[START:END]
        predictions = solve(openai, train_examples, test_examples)
        out = pd.DataFrame()
        out["qids"] = test_examples["question_id"].values
        out["gpt3"] = predictions
        out["question"] = test_examples["question"].values
        out["gold"] = test_examples["answer"].values
        out.to_csv(f"final_outputs/gpt3/{split}/gpt3_predictions{END}.csv")


def create_prompt_per_example(row, include_answer=False):
    """
    Creates a prompt for GPT-3
    """
    context = row["caption"]
    question = row["question"]
    gold = row["answer"]
    c = f"Context: {context}\n"
    q = f"Q: {question} "
    a = f"A: {gold}\n\n" if include_answer else "A:"

    return c + q + a


def solve(openai, train_df, test_df):
    """
    Train GPT-3 with few-shot learning
    :param openai: the OpenAI API object
    :param train_examples: list of dicts {"narrative", "plausible"}
    :param test_examples: list of dicts {"narrative", "plausible"}
    :return: the predictions
    """
    prompt = "Please answer the question according to the above context.\n"
    for index, row in train_df.iterrows():
        prompt += create_prompt_per_example(row, include_answer=True)
    predictions = []
    with tqdm(total=test_df.shape[0]) as pbar:

        for index, row in test_df.iterrows():
            gpt3_prompt = f"{prompt}{create_prompt_per_example(row, include_answer=False)}"
            pbar.update(1)
            print(gpt3_prompt)

            response = openai.Completion.create(
                engine="text-davinci-001",
                prompt=gpt3_prompt,
                temperature=0.7,
                max_tokens=10,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=["\n"]
            )
            pred = response['choices'][0]['text'].strip()
            predictions.append(pred)
            time.sleep(20)
    return predictions


def json_conv(filename):
    train_df = pd.read_csv(filename)
    train_df["qids"] = train_df["qids"].astype(str)
    train_df.fillna('', inplace=True)
    keys = list(train_df["qids"].values)
    vals = list(train_df["gpt3"].values)
    d = dict(zip(keys, vals))
    save_json('final_outputs/gpt3/train2014_gpt3.json', d)

    p = 'final_outputs/gpt3/val2014/gpt3_predictions_val2014.csv'
    v = pd.read_csv(p)
    v.fillna('', inplace=True)
    v["qids"] = v["qids"].astype(str)
    keys = list(v["qids"].values)
    vals = list(v["gpt3"].values)
    d = dict(zip(keys, vals))
    save_json('final_outputs/gpt3/val2014_gpt3.json', d)


if __name__ == "__main__":
    main()