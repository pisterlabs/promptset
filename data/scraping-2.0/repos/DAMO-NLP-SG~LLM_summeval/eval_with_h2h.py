import json
import os
from tqdm import tqdm
from prompt_templates import prepare_comp_prompt_mcq
import time
import argparse
# TODO: Important: enable your own openai key
import openai
from secret import my_key
openai.api_key = my_key


###### const #########
model_pairs = [("M22","M23"), ("M23","M17"),("M17","M12"),("M12","M13"), ("M13","M15"), ("M15","M14"), ("M14","M8"),("M8","M9"), ("M9","M10"), ("M10","M20"),("M20","M11")]
source_data_dir = "comp_data"
eval_root_dir = "comp_res"
if not os.path.exists(eval_root_dir):
    os.mkdir(eval_root_dir)

id2dim = {
    0:"relevance",
    1:"consistency",
    2:"fluency",
    3:"coherence"
}

########### helper functions ############

def parse_arguments(parser):
    ###Eval Hyperparameters
    # NOTE: "gpt-3.5-turbo-0301" may be deprecated, change to latest api model
    # gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301
    parser.add_argument('--eval_model', type=str, default="gpt-3.5-turbo-0301", help="the ChatGPT model to use")
    parser.add_argument('--dim', type=int, default=0, choices = [0,1,2,3], help="the evaluated dimension, see id2dim for conversions")
    parser.add_argument('--start_idx', type=int, default=0, help="evaluated example line start index, don't change unless need to rerun due to chatgpt gives error half way...")
    parser.add_argument('--end_idx', type=int, default=100, help="evaluated example line end index, don't change unless need to rerun due to chatgpt gives error half way...")

    parser.add_argument('--print_full_prompt_without_calling_api', action="store_true", default=False,
                        help="print the full prompt for each example")

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args

def call_api(aspect_id, summary, summary2, article, toggle, print_only=False, eval_max_len=128):
    # send request
    if toggle:
        eval_prompt = prepare_comp_prompt_mcq(aspect_id, summary2, summary, article)
    else:
        eval_prompt = prepare_comp_prompt_mcq(aspect_id, summary, summary2, article)

    # double-check full prompt before calling api
    if print_only:
        print(f"prompt:\n{eval_prompt}")
        exit()

    eval_msg = [
        {"role": "user", "content": eval_prompt},
    ]
    try:
        response = openai.ChatCompletion.create(
            model=eval_model, # gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301
            messages=eval_msg,
            temperature=0,
            max_tokens=eval_max_len,
        )
    except Exception as e:
        print("openai experiencing high volume, wait 20s to retry for 1st time...")
        time.sleep(20)
        try:
            response = openai.ChatCompletion.create(
                model=eval_model, # gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301
                messages=eval_msg,
                temperature=0,
                max_tokens=eval_max_len,
            )
        except Exception as e:
            print("openai experiencing high volume, wait 20s to retry for 2nd time...")
            time.sleep(20)
            response = openai.ChatCompletion.create(
                model=eval_model, # gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301
                messages=eval_msg,
                temperature=0,
                max_tokens=eval_max_len,
            )


    model_resp = response["choices"][0]["message"]["content"]
    prompt_len = response["usage"]["prompt_tokens"]
    total_len = response["usage"]["total_tokens"]
    print(model_resp)
    return (prompt_len, total_len, model_resp)

def main():
    parser = argparse.ArgumentParser()
    config = parse_arguments(parser)
    start_idx = config.start_idx
    end_idx = config.end_idx
    dim_id = config.dim
    eval_model = config.eval_model
    eval_dir = os.path.join(eval_root_dir, eval_model)
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)
    print_only = True if config.print_full_prompt_without_calling_api else False

    for model_1, model_2 in model_pairs:
        for toggle in [True, False]:
            dim = id2dim[dim_id]
            print(f"eval {model_1}-{model_2}, {dim}, toggle {toggle}")
            postfix = "_h2h"
            if toggle:
                postfix += "_toggle"

            with open(os.path.join(source_data_dir, model_1+"-"+model_2+"_"+dim+".txt"), 'r', encoding='utf-8') as f, \
                open(os.path.join(eval_dir, model_1+"-"+model_2+"_"+dim+postfix+".result"), "a", encoding="utf-8") as fd:
                dataset = [json.loads(line) for line in f]

                for i in tqdm(range(start_idx, end_idx)):
                    item = dataset[i]
                    id = item["id"]
                    source = item["src"]
                    summ1 = item["summary_1"]
                    summ2 = item["summary_2"]
                    score1 = item["human_score_1"]
                    score2 = item["human_score_2"]
                    
                    prompt_len, total_len, resp = call_api(dim_id, summ1, summ2, source, toggle, print_only)
                    
                    res = {
                        "id": id,
                        "prompt_len": prompt_len,
                        "total_len": total_len,
                        "resp":resp,
                        "human_score_1": score1,
                        "human_score_2": score2,
                    }
                    fd.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()