import os
import re
import json
from datetime import datetime

import guidance
import pandas as pd
import instructions as module_instruction
from common import load_api_key, load_cmd_line
from string import Template
import sys
sys.path.append("../")
from instructions import meta_instruction, sample_temp, prompt_temp, best_prompt_temp
from utils import seed_everything
from rec_utils import run_recommender
from templates import gpt_template


def build_sample(history, candidate, answer, click, **kwargs):
    full_prompt = Template(kwargs["prompt_temp"]).safe_substitute(history=history, candidate=candidate)
    return Template(sample_temp).safe_substitute(full_prompt=full_prompt, answer=answer, click=click)


def response_text(openai_resp):
    return openai_resp['choices'][0]['message']['content']


def is_descending(lst):
    for i in range(len(lst) - 1):
        if lst[i] < lst[i + 1]:
            return False
    return True


def cal_avg_performance(result):
    return round(result[["group_auc", "mean_mrr", "ndcg_5", "ndcg_10"]].mean(axis=1)[0], 2)


def update_monitor_scores(monitor_records, monitor_path, monitor_columns):
    monitor_scores = pd.DataFrame.from_records(monitor_records, columns=monitor_columns)
    if os.path.exists(monitor_path):
        monitor_scores = pd.concat([monitor_scores, pd.read_csv(monitor_path, index_col=False)])
    monitor_scores.drop_duplicates(subset=["epoch", "prompt_template", "group_auc"], inplace=True)
    monitor_scores.to_csv(monitor_path, index=False)
    return monitor_scores


def generate_improved_prompt(prompt4opt, model):
    return guidance(prompt4opt, llm=model, silent=True)()["output"]


def build_prompt(guide_instruction, initial_prompt, best_prompt, observe_instruction, samples, **kwargs):
    sample_num = kwargs.get('sample_num', 1)
    temperature = kwargs.get("temperature", 0)
    max_tokens = kwargs.get("max_tokens", 2048)
    prompt4opt = guide_instruction + Template(prompt_temp).safe_substitute(prompt_temp=initial_prompt)
    for line in samples.sample(n=sample_num, random_state=42).to_dict(orient="records"):
        prompt4opt += build_sample(prompt_temp=initial_prompt, history=line["history"], answer=line["output"],
                                   candidate=line["candidate"], click=line["label"])
    prompt4opt += Template(best_prompt_temp).safe_substitute(best_prompt_temp=best_prompt)
    prompt4opt += observe_instruction
    prompt4opt = Template(gpt_template).safe_substitute({
        "prompt_temp": prompt4opt, "temperature": temperature, "max_tokens": max_tokens,
        "system_instruction": meta_instruction
    })
    return prompt4opt


def build_optimizer(**kwargs):
    recommender = kwargs.get("recommender", "gpt-3.5-turbo-1106")
    optimizer = kwargs.get("optimizer", "gpt-3.5-turbo-1106")
    sample_num = kwargs.get('sample_num', 1)
    valid_samples = kwargs.get("samples", pd.read_csv(f"valid/sample100by_ratio.csv"))
    llm_seed = kwargs.get("llm_seed", 42)
    valid_samples = valid_samples.sample(n=kwargs.get("valid_sample_num", 100), random_state=42)
    middle_name = f"{recommender}--{optimizer}--{kwargs.get('tag', 'naive-format')}--sample_num-{sample_num}"
    print(middle_name)
    initial_prompt = getattr(module_instruction, kwargs.get("initial_prompt", "initial_prompt"))
    generated_output_path = f"generated_data/prompt_tuning/{middle_name}/epoch_0.csv"
    score_path = f"result/prompt_tuning/{middle_name}/epoch_0.csv"
    initial_result = run_recommender(initial_prompt, recommender=recommender, epoch=0, llm_seed=llm_seed,
                                     generated_output_path=generated_output_path, score_path=score_path)
    avg_performance = cal_avg_performance(initial_result)
    prompt_optimizer = guidance.llms.OpenAI(optimizer, api_key=load_api_key(), chat_mode=True)
    record = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 0, initial_prompt] + initial_result[[
        "group_auc", "mean_mrr", "ndcg_5", "ndcg_10"]].loc[0].tolist() + [avg_performance]
    monitor_columns = [
        "id", "epoch", "prompt_template", "group_auc", "mean_mrr", "ndcg_5", "ndcg_10", "avg_performance"
    ]
    monitor_records = [record]
    guide_instruction = getattr(module_instruction, kwargs.get("guide_instruction", "guide_instruction_naive"))
    observe_instruction = getattr(module_instruction,
                                  kwargs.get("observation_instruction", "observation_instruction_naive"))
    monitor_path = f"result/prompt_tuning/{middle_name}/monitor_scores.csv"
    assistant_path = f"result/prompt_tuning/{middle_name}/assistant.json"
    monitor_scores = update_monitor_scores(monitor_records, monitor_path, monitor_columns)
    best_scores = {"epoch": 0, "best_performance": avg_performance, "best_prompt_temp": initial_prompt}
    assistant_prompts = {}
    temperature = kwargs.get("temperature", 0)
    max_tokens = kwargs.get("max_tokens", 2048)
    for epoch in range(0, kwargs.get("epochs", 5)):
        generated_output_path = f"generated_data/prompt_tuning/{middle_name}/epoch_{epoch}.csv"
        samples = kwargs.get("samples", pd.read_csv(generated_output_path))
        prompt_params = {
            "guide_instruction": guide_instruction, "initial_prompt": initial_prompt, "best_prompt": initial_prompt,
            "observe_instruction": observe_instruction, "samples": samples, "temperature": temperature,
            "max_tokens": max_tokens , "sample_num": sample_num
        }
        prompt4opt = build_prompt(**prompt_params)
        go_on = True
        current_prompt = generate_improved_prompt(prompt4opt, prompt_optimizer)
        assistant_prompts[f"epoch-{epoch+1}"] = {
            "optimizer prompt": prompt4opt, "assistant answer": current_prompt,
        }
        while go_on:
            with open(assistant_path, "w") as f:
                json.dump(assistant_prompts, f, indent=4)
            match = re.search(r"# Prompt Template Begin\n(.*)# Prompt Template End", current_prompt, re.DOTALL)
            if match is not None:
                current_prompt = match.group(1)
                is_exist = False
                for epoch_no, prompts in assistant_prompts.items():
                    if "prompt template" in prompts and current_prompt == prompts["prompt template"]:
                        is_exist = True
                        break
                if not is_exist and "${history}" in current_prompt and "${candidate}" in current_prompt:
                    go_on = False
                else:  # request again
                    prompt4opt = build_prompt(**prompt_params)
                    current_prompt = generate_improved_prompt(prompt4opt, prompt_optimizer)
                    assistant_prompts[f"epoch-{epoch + 1}"]["assistant answer"] = current_prompt
            else:
                prompt4opt = build_prompt(**prompt_params)
                current_prompt = generate_improved_prompt(prompt4opt, prompt_optimizer)
                assistant_prompts[f"epoch-{epoch + 1}"]["assistant answer"] = current_prompt
        assistant_prompts[f"epoch-{epoch + 1}"]["prompt template"] = current_prompt
        with open(assistant_path, "w") as f:
            json.dump(assistant_prompts, f, indent=4)
        generated_output_path = f"generated_data/prompt_tuning/{middle_name}/epoch_{epoch+1}.csv"
        score_path = f"result/prompt_tuning/{middle_name}/epoch_{epoch+1}.csv"
        current_result = run_recommender(current_prompt, recommender=recommender, optimizer=optimizer, epoch=epoch+1,
                                         generated_output_path=generated_output_path, score_path=score_path,
                                         samples=valid_samples, llm_seed=llm_seed)
        avg_performance = cal_avg_performance(current_result)
        record = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch+1, current_prompt] + current_result[[
            "group_auc", "mean_mrr", "ndcg_5", "ndcg_10"]].loc[0].tolist() + [avg_performance]
        monitor_records.append(record)
        monitor_scores = update_monitor_scores(monitor_records, monitor_path, monitor_columns)
        if avg_performance > best_scores["best_performance"]:
            best_scores["epoch"] = epoch + 1
            best_scores["best_performance"] = avg_performance
            best_scores["best_prompt_temp"] = current_prompt
    with open(f"result/prompt_tuning/{middle_name}/best_scores.json", "w") as f:
        json.dump(best_scores, f, indent=4)
    return monitor_scores, best_scores


if __name__ == "__main__":
    seed_everything(42)
    args = load_cmd_line()
    build_optimizer(**args["cmd_args"])

