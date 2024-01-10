import os
import openai
import json
import pandas as pd
from tqdm import tqdm
openai.api_key = "OPENAI_API_KEY"

flag = "reclor"
response_list = {'context':[],'question':[],'optionA':[], 'optionB':[],'optionC':[],'optionD':[],'predict_answer':[]}
if flag == "reclor":
    with open("/data/qbao775/Logical-and-abstract-reasoning/data/ReClor/test.json", "r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    for item in tqdm(data):
        context = item["context"]
        question = item["question"]
        optionA = item["answers"][0]
        optionB = item["answers"][1]
        optionC = item["answers"][2]
        optionD = item["answers"][3]
        input_prompt = " Given context: " + context + " Question: " + \
            " A: " + optionA + " B: " + optionB + " C: " + optionC + \
            " D: " + optionD
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            # prompt=input_prompt,
            messages = [{"role": "system", "content" : "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible. Please only generate A, B, C or D as your predicted answer for the following input."},
            {"role": "user", "content" : "How are you?"},
            {"role": "assistant", "content" : "I am doing well"},
            {"role": "user", "content" : input_prompt}],
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        response_list["context"].append(context)
        response_list["question"].append(question)
        response_list["optionA"].append(optionA)
        response_list["optionB"].append(optionB)
        response_list["optionC"].append(optionC)
        response_list["optionD"].append(optionD)
        response_list["predict_answer"].append(response["choices"][0]["message"]["content"])
    df = pd.DataFrame(response_list)
    df.to_excel("chatgpt_reclor_prediction.xlsx")
        
        
elif flag == "logiqa":
    with open("/data/qbao775/Logical-and-abstract-reasoning/data/LogiQA/Test.json", "r", encoding="utf-8") as input_file:
        data = json.load(input_file)

    for item in tqdm(data):
        context = item["context"]
        question = item["question"]
        optionA = item["answers"][0]
        optionB = item["answers"][1]
        optionC = item["answers"][2]
        optionD = item["answers"][3]
        input_prompt = " Given context: " + context + " Question: " + \
            " A: " + optionA + " B: " + optionB + " C: " + optionC + \
            " D: " + optionD
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            # prompt=input_prompt,
            messages = [{"role": "system", "content" : "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible. Please only generate A, B, C or D as your predicted answer for the following input."},
            {"role": "user", "content" : "How are you?"},
            {"role": "assistant", "content" : "I am doing well"},
            {"role": "user", "content" : input_prompt}],
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        response_list["context"].append(context)
        response_list["question"].append(question)
        response_list["optionA"].append(optionA)
        response_list["optionB"].append(optionB)
        response_list["optionC"].append(optionC)
        response_list["optionD"].append(optionD)
        response_list["predict_answer"].append(response["choices"][0]["message"]["content"])
    df = pd.DataFrame(response_list)
    df.to_excel("chatgpt_logiqa_prediction.xlsx")
        
elif flag == "logiqav2":
    data = []
    with open("/data/qbao775/Logical-and-abstract-reasoning/data/LogiQA-V2/test.txt", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))

    for item in tqdm(data):
        context = item["text"]
        question = item["question"]
        optionA = item["options"][0]
        optionB = item["options"][1]
        optionC = item["options"][2]
        optionD = item["options"][3]
        input_prompt = " Given context: " + context + " Question: " + \
            " A: " + optionA + " B: " + optionB + " C: " + optionC + \
            " D: " + optionD
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            # prompt=input_prompt,
            messages = [{"role": "system", "content" : "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible. Please only generate A, B, C or D as your predicted answer for the following input."},
            {"role": "user", "content" : "How are you?"},
            {"role": "assistant", "content" : "I am doing well"},
            {"role": "user", "content" : input_prompt}],
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        response_list["context"].append(context)
        response_list["question"].append(question)
        response_list["optionA"].append(optionA)
        response_list["optionB"].append(optionB)
        response_list["optionC"].append(optionC)
        response_list["optionD"].append(optionD)
        response_list["predict_answer"].append(response["choices"][0]["message"]["content"])
    df = pd.DataFrame(response_list)
    df.to_excel("chatgpt_logiqav2_prediction.xlsx")