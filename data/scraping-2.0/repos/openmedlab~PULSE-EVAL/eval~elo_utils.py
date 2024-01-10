import glob
import json
import pandas as pd
import time
import os
import random
import re
from collections import defaultdict, OrderedDict
import regex as re
from datasets import load_dataset
from elo_analysis import report_elo_analysis_results
from collections import OrderedDict
import openai

mon = str(time.localtime().tm_mon)
mon = '0' + mon if len(mon) == 1 else mon
day = str(time.localtime().tm_mday)
day = '0' + day if len(day) == 1 else day
DATE = mon + day
    
    
def construct_elo_data(
    dataset_name,
    model_list,
    predict_dir="eval/predicted/",
    to_dir="eval/elo",
    start_p_idx=0,
    r_q_count = 10,
    seed=42,
    ):
    

    output_name = f"{dataset_name}_{DATE}.jsonl"
    print(f"Output data name: {output_name}")
    all_predict_paths = [predict_dir + model_name + f"/{dataset_name}.jsonl" for model_name in model_list]
    
    all_predict_data = OrderedDict()
    for path in all_predict_paths:
        model_name = path.split("/")[-2]
        all_predict_data[model_name] = []
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                line = line.rstrip()
                if len(line) == 0:
                    continue

                line = json.loads(line)
                all_predict_data[model_name].append(line)

    # check len
    d_len = None
    for item in all_predict_data.values():
        if d_len is None:
            d_len = len(item)
            
        assert d_len == len(item)

    # check q
    for idx in range(d_len):
        q = None
        for model_name in all_predict_data.keys():
            if q is None:
                q = all_predict_data[model_name][idx]['question']
            assert q == all_predict_data[model_name][idx]['question']

    all_predict_data = list(all_predict_data.items())
    print("Load predict success")
    
    random.seed(seed)
    round_data = []
    for a_idx in range(start_p_idx, len(all_predict_data)):
        for b_idx in range(0, a_idx):
            tmp_round_data = []
            while len(tmp_round_data) < r_q_count:
                d_idx = random.randint(0, d_len - 1)
                
                # # add shuffle
                if (a_idx + b_idx + d_idx) % 2 == 0:
                    ua_idx, ub_idx = a_idx, b_idx
                else:
                    ua_idx, ub_idx = b_idx, a_idx

                model_a = all_predict_data[ua_idx]
                model_b = all_predict_data[ub_idx]

                if model_a[1][d_idx]['predict_answer'] is None or len(model_a[1][d_idx]['predict_answer']) == 0:
                    continue
                
                if model_b[1][d_idx]['predict_answer'] is None or len(model_b[1][d_idx]['predict_answer']) == 0:
                    continue

                tmp_round_data.append({
                    # base
                    'type': model_a[1][d_idx]['type'],
                    'question': model_a[1][d_idx]['question'],
                    'reference_answer': model_a[1][d_idx]['reference_answer'],
                    # answer
                    "model_a": model_a[0],
                    "model_b": model_b[0],
                    "model_a_predict_answer": model_a[1][d_idx]['predict_answer'],
                    "model_b_predict_answer": model_b[1][d_idx]['predict_answer'],
                })
            
            round_data.extend(tmp_round_data)
            
    from collections import Counter

    print(Counter([item['model_a'] for item in round_data]))
    print(Counter([item['model_b'] for item in round_data]))
    print(f'New data length: {len(round_data)}')

    elo_inputs = []
    elo_prompt_ref = "[Question]\n{question}\n\n[Reference Answer]\n{reference_answer}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n[The End of Assistant 2's Answer]\n\n[System]\nWe would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nIf the answer does not give a direct option, score 0 points. If the answer gives the correct option, give 10 points. If the answer gives the wrong option, then give points appropriately based on his thought process.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."
    elo_prompt = ("[Question]\n{question}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n[The End of Assistant 2's Answer]\n\n[System]\nWe would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease read the question carefully, analyze the intention of the question, and then evaluate the quality of the responses.\nPlease rate the accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.")

    os.makedirs(os.path.join(to_dir, "elo_data"), exist_ok=True)
    with open(os.path.join(to_dir, "elo_data/"+output_name), "a", encoding="utf8") as f:
        for round_data_item in round_data:
            f.write(json.dumps(round_data_item, ensure_ascii=False) + "\n")
            
    for round_data_item in round_data:
        if round_data_item['reference_answer'] != '':
            content = elo_prompt_ref.format(
                        question=round_data_item['question'],
                        reference_answer=round_data_item['reference_answer'],
                        answer_1=round_data_item['model_a_predict_answer'],
                        answer_2=round_data_item['model_b_predict_answer'],
                    )
        else:
            content = elo_prompt.format(
                        question=round_data_item['question'],
                        answer_1=round_data_item['model_a_predict_answer'],
                        answer_2=round_data_item['model_b_predict_answer'],
                    )
            
        elo_inputs.append({
            "messages":[
                {
                    "role": "system", 
                    "content": "You are a helpful and precise assistant for checking the quality of the answer.",
                },
                {
                    "role": "user",
                    "content": content, 
                },   
            ],
            "functions":None,
        })

    os.makedirs(os.path.join(to_dir, "elo_inputs"), exist_ok=True)
    with open(os.path.join(to_dir, "elo_inputs/" + output_name), "a", encoding="utf8") as f:
        for elo_input in elo_inputs:
            f.write(json.dumps(elo_input, ensure_ascii=False) + "\n")
        print(f"Writed elo input and elo data files {output_name}.")
        
     
def parse_score(review):
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            raise Exception("Invalid score pair.")
    except Exception as e:
        print(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
        return None
   
    
def get_all_predict_data(
    all_predict_paths, 
    selected_models,
    elo_data_path,
    elo_outputs_path
    ):
    all_predict_data = {}
    for path in all_predict_paths:
        replace_pattern = re.compile(r"[_0-9]{0,10}.jsonl")
        dataset_name = re.sub(replace_pattern, "",  path.split("/")[-1])  # remove date

        elo_data = []
        elo_outputs = []
        
        with open(path, "r", encoding="utf8") as f_elo_data:
            for line in f_elo_data:
                line = line.rstrip()

                if len(line) == 0:
                    continue

                line = json.loads(line)

                elo_data.append(line)

        with open(path.replace(elo_data_path, elo_outputs_path), "r", encoding="utf8") as f_elo_outputs:

            for line in f_elo_outputs:
                line = line.rstrip()

                if len(line) == 0:
                    continue

                line = json.loads(line)['output']
                if line is None or len(line) == 0:
                    elo_outputs.append(None)

                else:
                    line = json.loads(line)
                    elo_outputs.append(parse_score(line['content']))
        
        assert len(elo_data) == len(elo_outputs)

        for elo_data_item, elo_output in zip(elo_data, elo_outputs):
            elo_data_item['score'] = elo_output

            if elo_output is None:
                elo_data_item['winner'] = "tie (None)"
            elif elo_output[0] > elo_output[1]:
                elo_data_item['winner'] = "model_a"
            elif elo_output[1] > elo_output[0]:
                elo_data_item['winner'] = "model_b"
            elif elo_output[0] == elo_output[1]:
                elo_data_item['winner'] = "tie"
            
        if dataset_name not in all_predict_data:
            all_predict_data[dataset_name] = []


        for elo_data_item in elo_data:
            if selected_models:
                if elo_data_item['model_a'] not in selected_models:
                    continue
                if elo_data_item['model_b'] not in selected_models:
                    continue
                
            all_predict_data[dataset_name].append(elo_data_item)
    return all_predict_data


def construct_elo_inputs(model_list, dataset_list, start_p_idx=0):
    for dataset in dataset_list:
        construct_elo_data(dataset, model_list=model_list, start_p_idx=start_p_idx)


def elo_evaluation(
    models, 
    datasets,
    elo_data_dir="eval/elo/elo_data"
    ):
    dataset_paths = []
    
    for dataset_name in datasets:
        dataset_paths += sorted(glob.glob(elo_data_dir + f"/{dataset_name}*.jsonl"))
    
    print("Start Elo evaluation:")
    elo_score_table = get_elo_rank(
        selected_models=models,
        all_predict_paths=dataset_paths,
    )
    
    return elo_score_table
 
    
def get_elo_rank(
    selected_models,
    all_predict_paths,
    elo_data_path = "eval/elo/elo_data/",
    elo_outputs_path = "eval/elo/elo_outputs/"
    ):
    
    all_predict_data = get_all_predict_data(
        selected_models=selected_models,
        all_predict_paths=all_predict_paths, 
        elo_data_path=elo_data_path,
        elo_outputs_path=elo_outputs_path,
    )
    
    elo_df = {}
    for task_name in all_predict_data.keys():
        tt_report = report_elo_analysis_results(all_predict_data[task_name])
        for k,v in tt_report['elo_rating_median'].items():
            if k not in elo_df:
                elo_df[k] = {item: None for item in all_predict_data.keys()}

            elo_df[k][task_name] = v

        print(task_name)
        try:
            tmp = {k:v for k, v in tt_report['elo_rating_median'].items()}
            print(tmp)
        except:
            print(tt_report['elo_rating_median'])
        print("----------------------------------------")
        
    all_data = []

    for v in all_predict_data.values():
        all_data.extend(v)
    print("ALL")
    all_report = report_elo_analysis_results(all_data)
    print("----------------------------------------")


    for k, v in all_report["elo_rating_median"].items():
        if k not in elo_df:
            elo_df[k] = {item: None for item in all_predict_data.keys()}
        elo_df[k]["ALL"] = v

    elo_df_use = pd.DataFrame(
        [
            {
            "Model Name": k, 
            "ALL": v["ALL"],
        **{dk:dv for dk,dv in v.items() if dk != "ALL"}
            } 
            for k,v in elo_df.items()
        ]
    )
    return elo_df_use


def GPT_generate(messages, model="gpt-4", stream=False):
    completion = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        stream=stream,
    )
    message = completion["choices"][0]["message"]
    return dict(message)

def api_evaluate(data):
    data['output'] = data["messages"]
    data['output'] = GPT_generate(data["messages"], )
    return data

def call_evaluator(
    api_func=api_evaluate,
    eval_date=None,
    in_dir="eval/elo/elo_inputs",
    to_dir="eval/elo/elo_outputs",
    num_proc=10,
):
    
    eval_date = DATE if not eval_date else eval_date
    eval_inputs = glob.glob(in_dir + f"/*{eval_date}.jsonl")
    for file_path in eval_inputs:
        dataset = load_dataset(
            "json", 
            data_files=file_path, 
            split="train"
        )
        
        eval_dataset = dataset.map(api_func, num_proc=num_proc, batched=False)
        
        to_path = file_path.replace(in_dir, to_dir)
        os.makedirs(to_path, exist_ok=True)
        with open(to_path, "a", encoding="utf-8") as writer:
            for data in eval_dataset:
                item = {
                    "input": {"messages": data["messages"]},
                    "output": data["output"],
                }
                writer.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Wrote elo evaluation results to {to_path}.")