import openai
from tqdm import tqdm
import json
from IPython import embed
import random
import subprocess
import re
from multiprocessing import Pool
from rouge_score import rouge_scorer
from functools import partial


key_pool = []
f = open("keys.txt", "r")
lines = f.readlines()
f.close()

for line in lines:
    key_pool.append(line.strip())

key_num = len(key_pool)


def chat_api(instr, current_key, temperature=0.3, sys_info="You are a helpful assistant to answer the question following instruction and demonstration examples."):
    print("~~~ In ChatGPT ~~~")
    # print(instr)
    try_limit = 50
    try_num = 0
    while try_num < try_limit:
        try_num += 1
        try:
            openai.api_key = key_pool[(try_num+current_key) % key_num]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": sys_info},
                    {"role": "user", "content": instr},
                ],
                temperature=temperature,
                max_tokens=2048,
            )
            return response["choices"][0]["message"]["content"]
        except:
            pass
    raise Exception("API key exhausted")


# ======================================================================= #

start_key = 0
temperature = 0.3
gen_func = chat_api

# ======================================================================== #

for test in ["dyck", "track_shuffle", "boolean", "date", "matrix", "arithmetic", "orientation", "remainder"]:
    want_prompt = True
    if want_prompt:
        f = open(f"../prompt_lib/prompt_cot/{test}_cot.txt", "r")
        prompt = f.read().strip()
        f.close()

    test_file = f"{test}_cot"
    save_path = f"../results_ChatGPT/test_cot/{test_file}.md"
    data_path = f"../datasets/test_cot/{test_file}.jsonl"

    f = open(data_path, "r")
    lines = f.readlines()
    f.close()

    if want_prompt:
        f = open(save_path, "w")
        f.write(prompt + "\n\n==============================prompt above! begin now!=====================================\n\n")
        f.close()
    else:
        f = open(save_path, "w")
        f.write(
            "==============================no prompt! begin now!=====================================\n\n")
        f.close()

    good = 0
    bad = 0
    for line in lines:
        data = json.loads(line.strip())
        env = prompt.strip() + "\n\n" + data["prompt"].strip()
        res = gen_func(env, start_key, temperature)
        model_res = res.split("###")[0].strip()

        std_ans = data["answer"]

        # print("================prompt=================")
        # print(data["prompt"])
        # print("===============model res==============")
        # print(model_res)
        
        f = open(save_path, "a")
        f.write(f"{data['prompt']}\n")
        f.write(f"=====model res=====\n{model_res}\n")
        f.write(str(std_ans).strip())
        
        success = False
        try:
            model_ans = model_res.split("Final Answer:")[-1].strip()
            if test in ["arithmetic", "remainder"]:
                ans_num = re.findall(r'-?\d+\.?\d*', model_ans)
                ans_num = [float(x) for x in ans_num]
                for ans in ans_num:
                    if round(ans, 2) == round(float(std_ans), 2):
                        good += 1
                        success = True
                        break
            elif test in ["orientation", "date"]:
                if str(std_ans).strip() in model_ans.strip():
                    good += 1
                    success = True
            elif test in ["dyck", "track_shuffle", "boolean"]:
                if model_ans.strip() != "" and model_ans.strip() in str(std_ans).strip():
                    good += 1
                    success = True
            elif test in ["matrix"]:
                ans_lists = re.findall(r"\[(.*?)\]", model_ans, re.S)
                for ans in ans_lists:
                    ans = eval("[" + ans + "]")
                    if ans == eval(str(std_ans)):
                        good += 1
                        success = True
                        break
            if not success:
                bad += 1
        except:
            bad += 1

        f = open(save_path, "a")
        f.write("\n=== std ans ===\n")
        f.write(str(std_ans).strip())
        if success:
            f.write("\nCorrect Answer!\n")
            print("Correct!")
        else:
            f.write("\nWrong Answer!\n")
            print("Wrong!")
        f.close()

        f = open(save_path, "a")
        f.write(
            f"\n\n==============================split line===================================\n\n")
        f.close()

    f = open(save_path, "a")
    f.write(f"good answer: {str(good)}\n")
    f.write(f"bad answer: {str(bad)}\n")
    avg = good / (good + bad)
    f.write(f"Average: {str(avg)}\n")
    f.close()
