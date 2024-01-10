import random
import time

import openai
import re

import pandas as pd

from truthfulqa import utilities

from reorder_realtoxicityprompts_02 import OPENAI_KEY
openai.api_key = OPENAI_KEY


def generate_exchange_first_last(dataset_file, w_file):
    dataset = utilities.load_questions(filename=dataset_file)
    # iterate through the dataset
    with open(w_file, 'w', encoding='utf-8') as f:
        for index, row in dataset.iterrows():
            prompt = row['Question']
            q_list = re.findall(r'\b\w+\b|[^\w\s]', prompt)
            q_list = q_list[-2] + ' ' + ' '.join(q_list[1:-2]) + ' ' + q_list[0] + ' ' + q_list[-1]
            row['Question'] = q_list
    utilities.save_questions(dataset, w_file)


def generate_random_two(dataset_file, w_file):
    dataset = utilities.load_questions(filename=dataset_file)
    with open(w_file, 'w', encoding='utf-8') as f:
        for index, row in dataset.iterrows():
            prompt = row['Question']
            q_list = re.findall(r'\b\w+\b|[^\w\s]', prompt)

            # exhange random two words in q_list
            i1 = random.randint(0, len(q_list) - 1)
            i2 = random.randint(0, len(q_list) - 1)
            t = q_list[i1]
            q_list[i1] = q_list[i2]
            q_list[i2] = t

            row['Question'] = ' '.join(q_list)
    utilities.save_questions(dataset, w_file)


def generate_exchange_adj_words(dataset_file, w_file):
    dataset = utilities.load_questions(filename=dataset_file)
    with open(w_file, 'w', encoding='utf-8') as f:
        for index, row in dataset.iterrows():
            prompt = row['Question']
            q_list = re.findall(r'\b\w+\b|[^\w\s]', prompt)

            new_i = []
            for i in range(len(q_list) // 2):
                t = [2 * i + 1, 2 * i]
                new_i.extend(t)
            if len(q_list) % 2 == 1:
                new_i.append(len(q_list) - 1)
            q_list = ' '.join([q_list[i] for i in new_i])

            row['Question'] = ' '.join(q_list)
    utilities.save_questions(dataset, w_file)


def generate_fix_first_last(dataset_file, w_file):
    dataset = utilities.load_questions(filename=dataset_file)
    with open(w_file, 'w', encoding='utf-8') as f:
        for index, row in dataset.iterrows():
            prompt = row['Question']
            q_list = re.findall(r'\b\w+\b|[^\w\s]', prompt)

            t2 = q_list[1:-1]
            random.shuffle(t2)
            q_list = q_list[0] + ' ' + ' '.join(t2) + ' ' + q_list[-1]

            row['Question'] = ' '.join(q_list)
    utilities.save_questions(dataset, w_file)


def reorder_truthfulQA(r1_file, r2_file, w_file, model_engien='gpt-3.5-turbo', start=0, end=100):
    dataset1 = utilities.load_questions(filename=r1_file)
    dataset1 = dataset1[start: end]
    dataset2 = utilities.load_questions(filename=r2_file)
    with open(w_file, 'w', encoding='utf-8') as f:
        for index, row in dataset1.iterrows():
            raw_prompt = row['Question']
            scramble_prompt = dataset2.loc[index, 'Question']

            answer = openai.ChatCompletion.create(
                model=model_engien,
                messages=[{"role": "system",
                           "content": "It is a query with scrambled word order, "
                                      "please reorder its words in a correct word order. "
                                      "You mustn't remove or add words, and the length should be kept. "
                                      },
                          {"role": "user", "content": "The query is %s"%scramble_prompt}],
                max_tokens=256, ).choices[0]['message']['content']
            time.sleep(4)
            print(answer)
            # 1024 for RealToxicityPrompt, completion
            # 256 for TruthfulQA, QA
            dataset1.loc[index, 'scrambled'] = scramble_prompt
            dataset1.loc[index, 'reordered'] = answer
    utilities.save_questions(dataset1, w_file)


def compute_match(r_file):
    datas =  utilities.load_questions(filename=r_file)
    original_prompt = datas['Question'].tolist()
    scrambled_prompt = datas['scrambled'].tolist()
    reordered_prompt = datas['reordered'].tolist()

    total = len(original_prompt)
    right_count, match_count = 0, 0

    for origin, scramble, reorder in zip(original_prompt, scrambled_prompt, reordered_prompt):
        origin = re.findall(r'\b\w+\b|[^\w\s]', origin)
        scramble = re.findall(r'\b\w+\b|[^\w\s]', scramble)
        reorder = re.findall(r'\b\w+\b|[^\w\s]', reorder)

        original_dict = {}
        for x in origin:
            if x not in original_dict:
                original_dict[x] = 1
            else:
                original_dict[x] += 1
        scrambled_dict = {}
        for x in scramble:
            if x not in scrambled_dict:
                scrambled_dict[x] = 1
            else:
                scrambled_dict[x] += 1
        reorder_dict = {}
        for x in reorder:
            if x not in reorder_dict:
                reorder_dict[x] = 1
            else:
                reorder_dict[x] += 1
        # whether the two dict is the same
        if original_dict == scrambled_dict:
            right_count += 1
        if original_dict == reorder_dict:
            match_count += 1
    print("total: %d, scrambled acc: %d, exact follow accuracy: %.4f"
          %(total, right_count/total, match_count/total))


def compute_metric(r_file, metric='meteor'):
    datas =  utilities.load_questions(filename=r_file)
    original_prompt = datas['Question'].tolist()
    scrambled_prompt = datas['scrambled'].tolist()
    reordered_prompt = datas['reordered'].tolist()

    import evaluate
    metrics = evaluate.load(metric)
    metric_score_list, origin_score_list = [], []
    for origin, _, reorder in zip(original_prompt, scrambled_prompt, reordered_prompt):
        if metric in ['bleurt']:
            metric_score = metrics.compute(references=[origin], predictions=[reorder])
            metric_score_origin = metrics.compute(references=[origin], predictions=[origin])
            metric_score_list.append(metric_score['scores'][0])
            origin_score_list.append(metric_score_origin['scores'][0])
        elif metric in ['rouge']:
            metric_score = metrics.compute(references=[origin], predictions=[reorder])
            metric_score_origin = metrics.compute(references=[origin], predictions=[origin])
            metric_score_list.append(metric_score['rouge1'])
            origin_score_list.append(metric_score_origin['rouge1'])
        else:
            metric_score = metrics.compute(references=[[origin]], predictions=[reorder])
            metric_score_origin = metrics.compute(references=[[origin]], predictions=[origin])
            metric_score_list.append(metric_score[metric])
            origin_score_list.append(metric_score_origin[metric])
    print("The avg %s score is: %s (%s in original)"%(metric, sum(metric_score_list) / len(metric_score_list), sum(origin_score_list) / len(origin_score_list)))



if __name__ == "__main__":
    origin_file = "../TruthfulQA/TruthfulQA_100.csv"

    exchange_first_last_file = "../TruthfulQA/TruthfulQA_100_exchange_first_last.csv"
    exchange_random_two_file = "../TruthfulQA/TruthfulQA_100_exchange_random_two.csv"

    exchange_adj_words_file = "../TruthfulQA/TruthfulQA_100_exchange_adj_words.csv"
    fix_first_last_file = "../TruthfulQA/TruthfulQA_100_fix_first_last.csv"

    # step1: generate scrambled orders
    # generate_exchange_first_last(dataset_file=origin_file, w_file=exchange_first_last_file)
    # generate_random_two(dataset_file=origin_file, w_file=exchange_random_two_file)
    # generate_exchange_adj_words(dataset_file=origin_file, w_file=exchange_adj_words_file)
    # generate_fix_first_last(dataset_file=origin_file, w_file=fix_first_last_file)

    # step2: reorder the scrambled orders
    # reorder_truthfulQA(r1_file=origin_file, r2_file=exchange_first_last_file, w_file="../TruthfulQA/TruthfulQA_reorder_exchange_first_last.csv", model_engien='gpt-3.5-turbo', start=70, end=100)
    # reorder_truthfulQA(r1_file=origin_file, r2_file=exchange_random_two_file, w_file="../TruthfulQA/TruthfulQA_reorder_exchange_random_two.csv", model_engien='gpt-3.5-turbo', start=70, end=100)
    # reorder_truthfulQA(r1_file=origin_file, r2_file=exchange_adj_words_file, w_file="../TruthfulQA/TruthfulQA_reorder_exchange_adj_words.csv", model_engien='gpt-3.5-turbo', start=70, end=100)
    # reorder_truthfulQA(r1_file=origin_file, r2_file=fix_first_last_file, w_file="../TruthfulQA/TruthfulQA_reorder_fix_first_last.csv", model_engien='gpt-3.5-turbo', start=70, end=100)

    # step3: Answering with the scrambled orders prompts, also evaluating.
    # see evaluate.py

    # step4: evalute the reordered ones
    # compute_match(r_file="../TruthfulQA/TruthfulQA_reorder_exchange_first_last.csv")
    # compute_match(r_file="../TruthfulQA/TruthfulQA_reorder_exchange_random_two.csv")
    # compute_match(r_file="../TruthfulQA/TruthfulQA_reorder_exchange_adj_words.csv")
    # compute_match(r_file="../TruthfulQA/TruthfulQA_reorder_fix_first_last.csv")

    compute_metric(r_file="../TruthfulQA/TruthfulQA_reorder_exchange_first_last.csv", metric='rouge')
    compute_metric(r_file="../TruthfulQA/TruthfulQA_reorder_exchange_random_two.csv", metric='rouge')
    compute_metric(r_file="../TruthfulQA/TruthfulQA_reorder_exchange_adj_words.csv", metric='rouge')
    compute_metric(r_file="../TruthfulQA/TruthfulQA_reorder_fix_first_last.csv", metric='rouge')
