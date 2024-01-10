import argparse
import json
import os
from tqdm import tqdm
from prompt_templates import prepare_rts_prompt, prepare_mcq_prompt, prepare_stareval_prompt
import time
# TODO: Important: enable your own openai key
import openai
from secret import my_key
openai.api_key = my_key


###### const #########
M_ID_LIST = ["M8","M9","M10","M11","M12","M13","M14","M15","M17","M20","M22","M23"]
annotation_dir = "model_output_annotations"
eval_root_dir = "eval_model_generations"
if not os.path.exists(eval_root_dir):
    os.mkdir(eval_root_dir)


########### helper functions ############

def parse_arguments(parser):
    ###Eval Hyperparameters
    # NOTE: "gpt-3.5-turbo-0301" may be deprecated, change to latest api model
    # gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301

    parser.add_argument('--eval_model', type=str, default="gpt-3.5-turbo-0301", help="the ChatGPT model to use")
    parser.add_argument('--dim', type=int, default=0, choices = [0,1,2,3], help="the evaluated dimension, see id2dim for conversions")
    parser.add_argument('--eval_type', type=int, default=0, choices = [0,1,2], help="evaluation method, 0 for rts, 1 for mcq, 2 for stareval")
    parser.add_argument('--start_idx', type=int, default=0, help="evaluated example line start index, don't change unless need to rerun due to chatgpt gives error half way...")
    parser.add_argument('--end_idx', type=int, default=100, help="evaluated example line end index, don't change unless need to rerun due to chatgpt gives error half way...")

    parser.add_argument('--print_full_prompt_without_calling_api', action="store_true", default=False,
                        help="print the full prompt for each example")

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args



id2dim = {
    0:"relevance",
    1:"consistency",
    2:"fluency",
    3:"coherence"
}

eval_types = {
    0: "reason", 
    1: "mcq", 
}

def call_api(eval_type_id, aspect_id, summary, article, results_dir, print_only=False, eval_max_len=128):
    # send request
    if eval_type_id == 0:
        eval_prompt = prepare_rts_prompt(aspect_id, summary, article)
    elif eval_type_id == 1:
        eval_prompt = prepare_mcq_prompt(aspect_id, summary, article)
    elif eval_type_id == 2:
        eval_prompt = prepare_stareval_prompt(aspect_id, summary, article)

    # double-check full prompt before calling api
    if print_only:
        print(f"prompt:\n{eval_prompt}")
        exit()

    eval_msg = [
        {"role": "user", "content": eval_prompt},
    ]
    try:
        response = openai.ChatCompletion.create(
            model= eval_model, 
            messages=eval_msg,
            temperature=0,
            max_tokens=eval_max_len,
        )
    except Exception as e:
        print("openai experiencing high volume, wait 10s to retry for 1st time...")
        time.sleep(10)
        try:
            response = openai.ChatCompletion.create(
                model= eval_model,
                messages=eval_msg,
                temperature=0,
                max_tokens=eval_max_len,
            )
            
        except Exception as e:
            print("openai experiencing high volume, wait 20s to retry for 2nd time...")
            time.sleep(20)
            response = openai.ChatCompletion.create(
                model= eval_model,
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
    eval_type_id = config.eval_type
    eval_dir = os.path.join(eval_root_dir, eval_model)
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)
    print_only = True if config.print_full_prompt_without_calling_api else False



    for M_ID in M_ID_LIST:
        with open(os.path.join(annotation_dir, M_ID+"_outputs_annotations.jsonl")) as f:
            print(M_ID)
            dataset = [json.loads(line) for line in f]
            # open files for score
            eval_results_dir = os.path.join(eval_dir,"eval_"+M_ID+"_generations")
            if not os.path.exists(eval_results_dir):
                os.mkdir(eval_results_dir)

            
            if eval_type_id == 0:
                postfix = "_rts"
            elif eval_type_id == 1:
                postfix = "_mcq"
            elif eval_type_id == 2:
                postfix = "_stareval"

            if dim_id == 0:
                f0 = open(os.path.join(eval_results_dir, id2dim[0]+postfix+".txt"),"a", encoding="utf-8")
                print("eval relevance")
                for i in tqdm(range(start_idx, end_idx)): 
                    example = dataset[i]
                    model = example['model_id']
                    assert model == M_ID
                    id = example['id']
                    summary = example['decoded']
                    article = example['text']
                    # get scores
                    prompt_len, total_len, resp = call_api(eval_type_id, dim_id, summary, article, eval_results_dir, print_only)
                    obj = {"id": id, "prompt_len":prompt_len, "total_len": total_len, "resp": resp}
                    f0.write(json.dumps(obj, ensure_ascii=False) + "\n")

            if dim_id == 1:
                f1 = open(os.path.join(eval_results_dir, id2dim[1]+postfix+".txt"),"a", encoding="utf-8")
                print("eval consistency")
                for i in tqdm(range(start_idx, end_idx)):
                    example = dataset[i]
                    model = example['model_id']
                    assert model == M_ID
                    id = example['id']
                    summary = example['decoded']
                    article = example['text']
                    prompt_len, total_len, resp = call_api(eval_type_id, dim_id, summary, article, eval_results_dir, print_only)
                    obj = {"id": id, "prompt_len":prompt_len, "total_len": total_len, "resp": resp}
                    f1.write(json.dumps(obj, ensure_ascii=False) + "\n")
                
            if dim_id == 2:
                f2 = open(os.path.join(eval_results_dir, id2dim[2]+postfix+".txt"),"a", encoding="utf-8")
                print("eval fluency") 
                for i in tqdm(range(start_idx, end_idx)):
                    example = dataset[i]
                    model = example['model_id']
                    assert model == M_ID
                    id = example['id']
                    summary = example['decoded']
                    article = example['text']
                    prompt_len, total_len, resp = call_api(eval_type_id, dim_id, summary, article, eval_results_dir, print_only)
                    obj = {"id": id, "prompt_len":prompt_len, "total_len": total_len, "resp": resp}
                    f2.write(json.dumps(obj, ensure_ascii=False) + "\n")

            if dim_id == 3:
                f3 = open(os.path.join(eval_results_dir, id2dim[3]+postfix+".txt"),"a", encoding="utf-8")
                print("eval coherence") 
                for i in tqdm(range(start_idx, end_idx)):
                    example = dataset[i]
                    model = example['model_id']
                    assert model == M_ID
                    id = example['id']
                    summary = example['decoded']
                    article = example['text']

                    prompt_len, total_len, resp = call_api(eval_type_id, dim_id, summary, article, eval_results_dir, print_only)
                    obj = {"id": id, "prompt_len":prompt_len, "total_len": total_len, "resp": resp}
                    f3.write(json.dumps(obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()