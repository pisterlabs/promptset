# Package 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
## import re

## progress bar
from tqdm import tqdm
from random import sample
## from sklearn.cluster import KMeans
import random
import os
import pickle
import openai
from scipy.stats import entropy
import sys
import time
##-----------------------------------------------------------

## Load the GSM8K MATH Datasets
sys.path.append("datasets/")
from GSM8k_Prompt import *

##-----------------------------------------------------------

## Load the GPT3 GPT4 Text-Davinci-003 models
sys.path.append("models/GPT")
from GPT_API_3 import *
from GPT_API_4 import *
from Text_Davinci_003 import * 

# openai api functions to conduct early stopping and short generation
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY
# demo cot 4-shot ICL GSM8K testset Q10 
def Davinci_openai_stop(prompt, stop_index):
    response = openai.Completion.create(
    model="text-davinci-002",prompt = prompt, temperature = 1,
    max_tokens=1024, top_p=1, frequency_penalty=0,
    presence_penalty=0,logprobs = 5,
    stop = stop_index)
    return response
def Davinci_openai(prompt):
    response = openai.Completion.create(
    model="text-davinci-002",prompt = prompt, temperature = 1,
    max_tokens=32, top_p=1, frequency_penalty=0,
    presence_penalty=0,logprobs = 5)
    return response


# function generate token distribution table df_result_majority
def entropy_df_generation(dic_result,n_round = 3):
    df = pd.DataFrame(columns=["token", 'logprob'])
    for i in range(n_round):
        result_i = dic_result[i]["choices"]
        if '{' in  result_i[0]["logprobs"]["tokens"]:
            index = result_i[0]["logprobs"]["tokens"].index('{')
        else:
            index = result_i[0]["logprobs"]["tokens"].index(' {')
        sample_token = result_i[0]["logprobs"]["tokens"][index+1]
        sample_token_logprob = result_i[0]["logprobs"]["token_logprobs"][index+1]
        df.loc[len(df)] = [sample_token, sample_token_logprob]  
        log_list = result_i[0]["logprobs"]["top_logprobs"][index+1]
        for key in list(log_list.keys()):
            df.loc[len(df)] = [key, log_list[key]]
        df["token"] = [i.replace(" ","") for i in df["token"]]

    df_result = pd.DataFrame(columns=["token", 'logprob'])
    for token in np.unique(df["token"]):
        df_result.loc[len(df_result)] = [token, np.mean(df[df["token"] == token]["logprob"])]
    df_result["prob"] = np.exp(df_result["logprob"])
    df_result["prob"] = df_result["prob"]/np.sum(df_result["prob"])
    df_result["logprob"] = np.log(df_result["prob"])
    df_result = df_result.sort_values(by = "prob", ascending = False).reset_index(drop=True)
    df_result["cumprob"] = np.cumsum(df_result["prob"])
    df_result_majority = df_result[df_result["cumprob"] < 1] 
    df_result_majority = df_result_majority[[i.replace(' ','').isnumeric() for i in df_result_majority["token"]]].reset_index(drop=True)
    df_result_majority["prob"] = np.exp(df_result_majority["logprob"])
    df_result_majority["prob"] = df_result_majority["prob"]/np.sum(df_result_majority["prob"])
    df_result_majority["cumprob"] = np.cumsum(df_result_majority["prob"])
    return df_result_majority

# function to sample enough answer to check the entropy
def answer_diversity(ans_prompt,n_round = 3):
    count = 0 
    dic_result = {}
    while count < n_round:
        index_front  = -100
        index_end = 100 
        ans_generation = Davinci_openai(ans_prompt)
        if ("{" in ans_generation["choices"][0]["logprobs"]["tokens"]):
            index_front =  ans_generation["choices"][0]["logprobs"]["tokens"].index("{")
        if (" {" in ans_generation["choices"][0]["logprobs"]["tokens"]):
            index_front =  ans_generation["choices"][0]["logprobs"]["tokens"].index(" {")

        if ("}" in ans_generation["choices"][0]["logprobs"]["tokens"]):
            index_end =  ans_generation["choices"][0]["logprobs"]["tokens"].index("}")
        if ("} " in ans_generation["choices"][0]["logprobs"]["tokens"]):
            index_end =  ans_generation["choices"][0]["logprobs"]["tokens"].index("} ")
        
        if index_end - index_front == 2: 
            dic_result[count] = ans_generation
            count += 1 
    return dic_result

# function to generate the entropy related dataframe and value
def entropy_check(i,step_reasoning_list, n_shot_num = 4, n_step_num = 3):
    # answer entropy check
    ans_prompt = n_shot_prompt_generator_GSM8K_reason(n_shot_num ,n_step_num ,GSM8K_test_df["question"][i], GSM8K_train_df_reason, False, len(step_reasoning_list), True) 

    for _ in step_reasoning_list:
        ans_prompt += '\n'
        ans_prompt += _["choices"][0]["text"]
    ans_prompt += "\nThe answer is"

    ## sample 
    dic_result = answer_diversity(ans_prompt)

    ## dataprocessing
    df_summary = entropy_df_generation(dic_result)

    ## entrpy check
    entropy_value = entropy(df_summary["prob"])
    print(entropy_value)
    return ans_prompt, dic_result, df_summary, entropy_value

from functools import reduce  # forward compatibility for Python 3
import operator

# function to get value from a nested dictionary
def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)

# function to set value in a nested dictionary
def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value


# function to calculate the change of belief
def belif_change_index(dist1, dist2):
    index = 0
    token_list = list(set(dist1["token"]) | set(dist2["token"]))
    for i in token_list:
        if (i in list(dist1["token"])) and (i in list(dist2["token"])):
            index1 = list(dist1["token"]).index(i)
            index2 = list(dist2["token"]).index(i)
            index += np.abs(dist1["prob"][index1] - dist2["prob"][index2])

        elif (i not in list(dist1["token"])) and (i in list(dist2["token"])):
            index2 = list(dist2["token"]).index(i)
            index += np.abs(dist2["prob"][index2])        
        else:
            index1 = list(dist1["token"]).index(i)
            index += np.abs(dist1["prob"][index1])     
    return index

# function to diversity the son node from parent node 
def iteration(question_index, tmp_book, entropy_level = 1):
    layer_number = int(tmp_book["layer"].split("{")[1][0])
    step_str = "Step{"+str(layer_number+1) + "}Num"
    tmp = {}  
    tmp[step_str] = 0
    entropy_score = 10 
    # intial prompt 
    count = 0 
    print("In Step " + str(layer_number) + "\n")

    entropy_value = tmp_book["Status"][-1]
    dist1 = tmp_book["Status"][2]
    prompt = tmp_book["prompt"]
    step_reasoning = tmp_book["step_reasoning"]
    prompt += step_reasoning['choices'][0]["text"]

    while (entropy_score > entropy_level) &  (count < 2):
        step_reasoning_list = tmp_book["step_reasoning_list"].copy()
        print("-----------------------------------------------------") 
        step_reasoning = Davinci_openai_stop(prompt,["Step "+str(layer_number+2)])
        step_reasoning_list.append(step_reasoning)
        print('-------------------------------------------')
        for i in step_reasoning_list:
            print(i['choices'][0]["text"])
        print('-------------------------------------------')
        print(step_reasoning['choices'][0]["text"])   
        ans1 = entropy_check(question_index,step_reasoning_list , n_shot, n_step)
        dist2 = ans1[2]
        tmp[count] = {}
        tmp[count]["prompt"] = prompt
        tmp[count]["Status"] = ans1
        tmp[count]["step_reasoning"] = step_reasoning 
        tmp[count]["step_reasoning_list"] = step_reasoning_list
        tmp[step_str] += 1
        tmp[count]["layer"] = step_str
        tmp[count]["belif"] = belif_change_index(dist1, dist2)
        print("Change of Belif: " + str(tmp[count]["belif"]))
        count += 1
        print("In " + str(layer_number+1) +   " Step Generate: "+str(count))
        entropy_score = ans1[-1]
    return tmp