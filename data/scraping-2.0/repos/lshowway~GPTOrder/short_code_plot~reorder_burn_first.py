import json
import random
import time

import openai
import re

import pandas as pd

from truthfulqa import utilities

from key import Chat_API
openai.api_key = Chat_API


def reorder_evals(r1_file, w_file, model_engien='gpt-3.5-turbo', start=0, end=100):
    dataset1 = []
    with open(r1_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            line = json.loads(line)
            dataset1.append(line)
    dataset1 = dataset1[0: end]
    if start == 0:
        t = 'w'
    else:
        t = 'a'
    with open(w_file, t, encoding='utf-8') as f:
        for i, row in enumerate(dataset1):
            if i < start:
                continue
            new_line = {}
            if 'coqa' in r1_file:
                original_prompt = row['origin_content']
            else:
                original_prompt = row['input'][-1]['origin_content']
            scrambled_prompt = row['input'][-1]['content']
            answer = openai.ChatCompletion.create(
                model=model_engien,
                messages=[{"role": "system",
                           "content": "It is a query with scrambled word order, "
                                          "please reorder its words in a correct word order. "
                                          "You mustn't remove or add words, and the length should be kept. "
                                          },
                          {"role": "user", "content": "The query is %s"%scrambled_prompt}],
                max_tokens=128, ).choices[0]['message']['content']
            time.sleep(20)
            print(answer)
            # 1024 for RealToxicityPrompt, completion
            # 256 for TruthfulQA, QA
            # 256 for Evals
            new_line['origin_prompt'] = original_prompt
            new_line['scrambled_prompt'] = scrambled_prompt
            new_line['reordered'] = answer
            json.dump(new_line, f)
            f.write('\n')


def compute_match(r_file):
    print(r_file)
    original_prompt, scrambled_prompt, reordered_prompt = [], [], []

    with open(r_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            original_prompt.append(line['origin_prompt'])
            scrambled_prompt.append(line['scrambled_prompt'])
            reordered_prompt.append(line['reordered'])

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
    print("total: %d, scrambled acc: %d, exact follow accuracy: %.4f"%(total, right_count/total, match_count/total))


def compute_metric(r_file, metric='meteor'):
    original_prompt, scrambled_prompt, reordered_prompt = [], [], []

    with open(r_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            original_prompt.append(line['origin_prompt'])
            scrambled_prompt.append(line['scrambled_prompt'])
            reordered_prompt.append(line['reordered'])

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
            metric_score = metrics.compute(references=[origin], predictions=[reorder])
            metric_score_origin = metrics.compute(references=[origin], predictions=[origin])
            metric_score_list.append(metric_score[metric])
            origin_score_list.append(metric_score_origin[metric])
    print("The avg %s score is: %s (%s in original)"%(metric, sum(metric_score_list) / len(metric_score_list), sum(origin_score_list) / len(origin_score_list)))


if __name__ == '__main__':
    # born_first
    r1_file = "/evals/evals/registry/data/born_first/born_first_exchange_first_last.jsonl"
    r2_file = "/evals/evals/registry/data/born_first/born_first_random_two.jsonl"
    r3_file = "/evals/evals/registry/data/born_first/born_first_exchange_adj.jsonl"
    r4_file = "/evals/evals/registry/data/born_first/born_first_fix_first_last.jsonl"

    w1 = "./output/born-first-exchange_first_last_reorder.jsonl"
    w2 = "./output/born-first-random_two_reorder.jsonl"
    w3 = "./output/born-first-exchange_adj_reorder.jsonl"
    w4 = "./output/born-first-fix_first_last_reorder.jsonl"

    # reorder_evals(r1_file, w1, model_engien='gpt-3.5-turbo', start=0, end=100)
    # reorder_evals(r2_file, w2, model_engien='gpt-3.5-turbo', start=0, end=100)
    # reorder_evals(r3_file, w3, model_engien='gpt-3.5-turbo', start=0, end=100)
    # reorder_evals(r4_file, w4, model_engien='gpt-3.5-turbo', start=0, end=100)

    # compute_match(r_file=w3)
    # compute_metric(r_file=w3, metric='meteor')
    # compute_metric(r_file=w3, metric='bleurt')
    # compute_metric(r_file=w3, metric='bleu')
    # compute_metric(r_file=w4, metric='rouge')

    # # task: logic
    # r1_file = "D:\phd5/UCPH\GPTOrder\evals\evals/registry/data/logic/samples_exchange_first_last.jsonl"
    # r2_file = "D:\phd5/UCPH\GPTOrder\evals\evals/registry/data/logic/samples_random_two.jsonl"
    # r3_file = "D:\phd5/UCPH\GPTOrder\evals\evals/registry/data/logic/samples_exchange_adj.jsonl"
    # r4_file = "D:\phd5/UCPH\GPTOrder\evals\evals/registry/data/logic/samples_fix_first_last.jsonl"

    # w1 = "./output/logic-fact-exchange_first_last_reorder"
    # w2 = "./output/logic-fact-random_two_reorder"
    # w3 = "./output/logic-fact-exchange_adj_reorder"
    # w4 = "./output/logic-fact-fix_first_last_reorder"

    # reorder_evals(r1_file, w1, model_engien='gpt-3.5-turbo', start=0, end=100)
    # reorder_evals(r2_file, w2, model_engien='gpt-3.5-turbo', start=0, end=100)
    # reorder_evals(r3_file, w3, model_engien='gpt-3.5-turbo', start=0, end=100)
    # reorder_evals(r4_file, w4, model_engien='gpt-3.5-turbo', start=0, end=100)

    # compute_match(r_file=w4)
    # compute_metric(r_file=w4, metric='meteor')
    # compute_metric(r_file=w4, metric='bleurt')
    # compute_metric(r_file=w4, metric='bleu')
    # compute_metric(r_file=w4, metric='rouge')

    # task: infinite
    r1_file = "/evals/evals/registry/data/infiniteloop-match/infiniteloop-match_exchange_first_last.jsonl"
    r2_file = "/evals/evals/registry/data/infiniteloop-match/infiniteloop-match_random_two.jsonl"
    r3_file = "/evals/evals/registry/data/infiniteloop-match/infiniteloop-match_adj.jsonl"
    r4_file = "/evals/evals/registry/data/infiniteloop-match/infiniteloop-match_fix_first_last.jsonl"

    w1 = "./output/infiniteloop-match-exchange_first_last_reorder.jsonl"
    w2 = "./output/infiniteloop-match-random_two_reorder.jsonl"
    w3 = "./output/infiniteloop-match-exchange_adj_reorder.jsonl"
    w4 = "./output/infiniteloop-match-fix_first_last_reorder.jsonl"

    # reorder_evals(r1_file, w1, model_engien='gpt-3.5-turbo', start=0, end=100)
    # reorder_evals(r2_file, w2, model_engien='gpt-3.5-turbo', start=87, end=100)
    # reorder_evals(r3_file, w3, model_engien='gpt-3.5-turbo', start=52, end=100)
    # reorder_evals(r4_file, w4, model_engien='gpt-3.5-turbo', start=0, end=100)

    # compute_match(r_file=w4)
    # compute_metric(r_file=w4, metric='meteor')
    # compute_metric(r_file=w4, metric='bleurt')
    # compute_metric(r_file=w4, metric='bleu')
    # compute_metric(r_file=w4, metric='rouge')

    # task: computer
    r1_file = "/evals/evals/registry/data/test_comp_sci/questions_exchange_first_last.jsonl"
    r2_file = "/evals/evals/registry/data/test_comp_sci/questions_random_two.jsonl"
    r3_file = "/evals/evals/registry/data/test_comp_sci/questions_exchange_adj.jsonl"
    r4_file = "/evals/evals/registry/data/test_comp_sci/questions_fix_first_last.jsonl"

    w1 = "./output/comp_sci-exchange_first_last_reorder.jsonl"
    w2 = "./output/comp_sci-random_two_reorder.jsonl"
    w3 = "./output/comp_sci-exchange_adj_reorder.jsonl"
    w4 = "./output/comp_sci-fix_first_last_reorder.jsonl"

    # reorder_evals(r1_file, w1, model_engien='gpt-3.5-turbo', start=0, end=100)
    # reorder_evals(r2_file, w2, model_engien='gpt-3.5-turbo', start=0, end=100)
    # reorder_evals(r3_file, w3, model_engien='gpt-3.5-turbo', start=36, end=100)
    # reorder_evals(r4_file, w4, model_engien='gpt-3.5-turbo', start=15, end=100)

    # compute_match(r_file=w4)
    # compute_metric(r_file=w4, metric='meteor')
    # compute_metric(r_file=w4, metric='bleurt')
    # compute_metric(r_file=w4, metric='bleu')
    # compute_metric(r_file=w4, metric='rouge')

    # task: coqa
    # r1_file = "D:\phd5/UCPH\GPTOrder\evals\evals/registry\data/coqa/samples_exchange_first_last.jsonl"
    # r2_file = "D:\phd5/UCPH\GPTOrder\evals\evals/registry\data/coqa/samples_random_two.jsonl"
    # r3_file = "D:\phd5/UCPH\GPTOrder\evals\evals/registry\data/coqa/samples_exchange_adj.jsonl"
    # r4_file = "D:\phd5/UCPH\GPTOrder\evals\evals/registry\data/coqa/samples_fix_first_last.jsonl"

    # w1 = "./output/coqa-exchange_first_last_reorder.jsonl"
    # w2 = "./output/coqa-random_two_reorder.jsonl"
    # w3 = "./output/coqa-exchange_adj_reorder.jsonl"
    # w4 = "./output/coqa-fix_first_last_reorder.jsonl"

    # reorder_evals(r1_file, w1, model_engien='gpt-3.5-turbo', start=0, end=100)
    # reorder_evals(r2_file, w2, model_engien='gpt-3.5-turbo', start=0, end=100)
    # reorder_evals(r3_file, w3, model_engien='gpt-3.5-turbo', start=0, end=100)
    # reorder_evals(r4_file, w4, model_engien='gpt-3.5-turbo', start=0, end=100)

    # compute_match(r_file=w4)
    # compute_metric(r_file=w4, metric='meteor')
    # compute_metric(r_file=w4, metric='bleurt')
    # compute_metric(r_file=w4, metric='bleu')
    # compute_metric(r_file=w4, metric='rouge')