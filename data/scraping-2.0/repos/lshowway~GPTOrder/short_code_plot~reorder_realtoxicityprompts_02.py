# RealToxicityPrompt dataset, order reconstruction
import json
import re

import openai
import pandas as pd

from short_code_plot.reorder_realtoxicityprompts import load_RealToxicityPrompt_jsonl, OPENAI_KEY


def reorder_RealToxicityPrompt(r1_file, r2_file, w_file, model_engien="gpt-3.5-turbo", start=0, end=100):
    if start == 0:
        paradigm = 'w'
    else:
        paradigm = 'a'
    openai.api_key = OPENAI_KEY
    origin_prompts_list = load_RealToxicityPrompt_jsonl(r1_file)
    scrambled_prompts_list = load_RealToxicityPrompt_jsonl(r2_file)
    with open(w_file, paradigm, encoding='utf-8') as fw:
        for origin, prompt in zip(origin_prompts_list[start:end], scrambled_prompts_list[start:end]):
            reorder = openai.ChatCompletion.create(
                model=model_engien,
                messages=[{"role": "system",
                           "content": "It is a query with scrambled word order, "
                                      "please reorder its words in a correct word order. "
                                      "You mustn't remove or add words, and the length should be kept. "
                                      },
                          {"role": "user", "content": "The query is: %s"%prompt}],
                max_tokens=256,).choices[0]['message']['content']
            # reorder is 256
            t = {"original_prompt": origin, "scrambled_prompt": prompt, "reordered_prompt": reorder}
            fw.write(json.dumps(t) + '\n')
            fw.flush()
            print(origin)
            print(prompt)
            print(reorder)
            print("=====================================")


def compute_match(r_file):
    datas = pd.read_json(r_file, lines=True)
    original_prompt = datas["original_prompt"].tolist()
    scrambled_prompt = datas["scrambled_prompt"].tolist()
    reordered_prompt = datas["reordered_prompt"].tolist()

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
    datas = pd.read_json(r_file, lines=True)
    original_prompt = datas["original_prompt"].tolist()
    scrambled_prompt = datas["scrambled_prompt"].tolist()
    reordered_prompt = datas["reordered_prompt"].tolist()

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


if __name__ == '__main__':
    origin_file = 'C:/Users/Lenovo/Desktop/realtoxicityprompts-data/realtoxicityprompts-data/manually_select_China_400.jsonl'
    o1_file = "C:/Users/Lenovo/Desktop/realtoxicityprompts-data/realtoxicityprompts-data/RealToxicityPrompt_china_exchange_first_last.jsonl"
    o2_file = "C:/Users/Lenovo/Desktop/realtoxicityprompts-data/realtoxicityprompts-data/RealToxicityPrompt_china_random_two.jsonl"
    o3_file = "C:/Users/Lenovo/Desktop/realtoxicityprompts-data/realtoxicityprompts-data/RealToxicityPrompt_china_exchange_adj_words.jsonl"
    o4_file = "C:/Users/Lenovo/Desktop/realtoxicityprompts-data/realtoxicityprompts-data/RealToxicityPrompt_china_fix_first_last.jsonl"

    # step 1: scrambled word order
    reorder_RealToxicityPrompt(r1_file=origin_file, r2_file=o1_file,
                               w_file="../short_code_main/output/RealToxicityPrompt_china_exchange_first_last_reorder.jsonl", start=11, end=100)
    # reorder_prompt(r1_file=origin_file, r2_file=o2_file,
                   # w_file="./output/RealToxicityPrompt_china_random_two_reorder.jsonl", start=62, end=100)
    # reorder_prompt(r1_file=origin_file, r2_file=o3_file,
    #                w_file="./output/RealToxicityPrompt_china_exchange_adj_words_reorder.jsonl", start=57, end=100)
    # reorder_prompt(r1_file=origin_file, r2_file=o4_file,
    #                w_file="./output/RealToxicityPrompt_china_fix_first_last_reorder.jsonl", start=99 , end=100)

    # step 2: compare the reordered and the original text
    # compute_match(r_file="./output/RealToxicityPrompt_china_exchange_first_last_reorder.jsonl")
    # compute_match(r_file="./output/RealToxicityPrompt_china_random_two_reorder.jsonl")
    # compute_match(r_file="./output/RealToxicityPrompt_china_exchange_adj_words_reorder.jsonl")
    # compute_match(r_file="./output/RealToxicityPrompt_china_exchange_fix_first_last_reorder.jsonl")

    # step 3: compute the metric
    # compute_metric(r_file="./output/RealToxicityPrompt_china_exchange_first_last_reorder.jsonl", metric='rouge')
    # compute_metric(r_file="./output/RealToxicityPrompt_china_random_two_reorder.jsonl", metric='rouge')
    # compute_metric(r_file="./output/RealToxicityPrompt_china_exchange_adj_words_reorder.jsonl", metric='rouge')
    # compute_metric(r_file="./output/RealToxicityPrompt_china_exchange_fix_first_last_reorder.jsonl", metric='rouge')