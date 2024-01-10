import json
import os
import re
import openai
import google.generativeai as palm
import pandas as pd

from tqdm import tqdm
from common import (
    chat,
    load_api_key,
    load_cmd_line,
    evaluate_one
)
from one_shot_example import one_shot
from utils import evaluate_performance


if __name__ == "__main__":
    openai.api_key = load_api_key()
    palm.configure(api_key=load_api_key("google_key.json"))
    args = load_cmd_line()
    model_name = args.get("model_name", "gpt-3.5-turbo")
    sample_num = args.get("sample_num", 1000)
    params = {
        "max_tokens": args.get("max_tokens", args.get("max_tokens", 30)),
    }
    temp_name = args.get("prompt_temp", "naive_zero_shot")
    prompt_temp = json.load(open("prompt_temp.json", "r"))[temp_name]
    mode = args.get("mode", "rank")
    sampled_df = pd.read_csv(f"sampled_{sample_num}.csv")
    max_num = min(args.get("max_num", 10), len(sampled_df))  # only request 10 samples by default
    saved_model_name = "palm" if model_name == "models/chat-bison-001" else model_name
    suffix = f"{max_num}_{saved_model_name}_{mode}_{temp_name}"
    os.makedirs("generated_data", exist_ok=True)
    os.makedirs("result", exist_ok=True)
    performance = []
    for index in tqdm(sampled_df.sample(max_num).index, total=max_num):
        hist, cand, label = sampled_df.loc[index, ["history", "candidate", "label"]]
        if mode == "explanation":
            full_prompt = prompt_temp.format(hist=hist, cand=cand)
        else:
            if "one_shot" in temp_name:
                full_prompt = prompt_temp.format(one_shot=one_shot, hist=hist, cand=cand)
            else:
                full_prompt = prompt_temp.format(hist=hist, cand=cand)
        try:
            output = chat(full_prompt, model=model_name, max_try=5, **params)
        except:
            output = ""
        print(output)
        sampled_df.loc[index, "prediction"] = ",".join(re.findall(r"C\d+", output))
        sampled_df.loc[index, model_name] = output
        sampled_df.to_csv(f"generated_data/sampled_{suffix}.csv", index=False)
        result = list(evaluate_one(label.split(","), sampled_df.loc[index, "prediction"]))
        performance.append([sampled_df.loc[index, "impression_id"]] + result)
        performance_df = evaluate_performance(performance)
        performance_df.to_csv(
            f"result/sampled_{suffix}.csv",
            index=False,
        )
