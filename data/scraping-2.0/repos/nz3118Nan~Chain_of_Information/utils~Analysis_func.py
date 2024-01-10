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


# function for data processing
def summary_extrator(dic_collection, dataset):
    dic_result_df = {}
    df_Q_summary = pd.DataFrame(columns=['question','answer_clean', 
                                'test_answer', 'answer_list', 'prob_list', 'entropy', 'index_distance', 'index'])
    n_round = 5
    for i in tqdm(range(len(dataset))):
        dic_result_i = dic_collection[i]
        df = pd.DataFrame(columns=["token", 'logprob'])
        index_distence = 0
        index_distence_list = []
        for j in range(n_round):
            result_i = dic_result_i[j]["choices"]
            if '{' in result_i[0]["logprobs"]["tokens"]:
                index = result_i[0]["logprobs"]["tokens"].index('{')
            else:
                index = result_i[0]["logprobs"]["tokens"].index(' {')

            if '}' in result_i[0]["logprobs"]["tokens"]:
                index_next = result_i[0]["logprobs"]["tokens"].index('}')
            elif ' }' in result_i[0]["logprobs"]["tokens"]:
                index_next = result_i[0]["logprobs"]["tokens"].index(' }')
            else:
                index_next = 100

            if (index+1 >= len(result_i[0]["logprobs"]["tokens"])) | (index_next - index != 2):
                continue 
            else:
                sample_token = result_i[0]["logprobs"]["tokens"][index+1]
                sample_token_logprob = result_i[0]["logprobs"]["token_logprobs"][index+1]
        
                df.loc[len(df)] = [sample_token, sample_token_logprob]

                log_list = result_i[0]["logprobs"]["top_logprobs"][index+1]

                for key in list(log_list.keys()):
                    df.loc[len(df)] = [key, log_list[key]]

                index_distence_list.append((index_next - index))

        if len(df) == 0:
            continue
        else:
            df_result = pd.DataFrame(columns=["token", 'logprob'])
            for token in np.unique(df["token"]):
                df_result.loc[len(df_result)] = [token, np.mean(df[df["token"] == token]["logprob"])]
            df_result["token"] = [i.replace(" ","") for i in df_result["token"]]
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

            df_Q_summary.loc[len(df_Q_summary)] = [dataset['question'][i], 
                                                    int(dataset['answer_clean'][i]),
                                                    int(df_result_majority["token"][0]),
                                                    [int(i) for i in df_result_majority["token"]],
                                                    list(df_result_majority["prob"]),
                                                    entropy(list(df_result_majority["prob"])),
                                                    np.mean(index_distence_list),
                                                    i]
    df_Q_summary["correct"] = list(df_Q_summary["answer_clean"] == df_Q_summary["test_answer"])
    print("Accuracy: " + str(np.round(np.sum(df_Q_summary["answer_clean"] == df_Q_summary["test_answer"])/len(df_Q_summary)*100,2)) + "%")  

    return df_Q_summary


# function for data processing
def summary_extrator_step(dic_collection, dataset, index_list):
    # data processing
    dic_result_df = {}
    df_Q_summary = pd.DataFrame(columns=['question','answer_clean', 
                                'test_answer', 'answer_list', 'prob_list', 'entropy', 'index_distance','index'])
    n_round = 5

    pbar = tqdm(total=len(index_list))

    for _ , i in enumerate(index_list):
        dic_result_i = dic_collection[i]
        df = pd.DataFrame(columns=["token", 'logprob'])
        index_distence = 0
        
        index_distence_list = []

        for j in range(n_round):
            result_i = dic_result_i[j]["choices"]
            if '{' in result_i[0]["logprobs"]["tokens"]:
                index = result_i[0]["logprobs"]["tokens"].index('{')
            else:
                index = result_i[0]["logprobs"]["tokens"].index(' {')

            if '}' in result_i[0]["logprobs"]["tokens"]:
                index_next = result_i[0]["logprobs"]["tokens"].index('}')
            elif ' }' in result_i[0]["logprobs"]["tokens"]:
                index_next = result_i[0]["logprobs"]["tokens"].index(' }')
            else:
                index_next = 100

            index_distence += (index_next - index)
            
            if (index+1 >= len(result_i[0]["logprobs"]["tokens"])) | (index_next - index > 2):
                continue 
            else:
                sample_token = result_i[0]["logprobs"]["tokens"][index+1]
                sample_token_logprob = result_i[0]["logprobs"]["token_logprobs"][index+1]
                
                df.loc[len(df)] = [sample_token, sample_token_logprob]

                log_list = result_i[0]["logprobs"]["top_logprobs"][index+1]

                for key in list(log_list.keys()):
                    df.loc[len(df)] = [key, log_list[key]]

                index_distence_list.append(index_next - index)

        
        if len(df) == 0:
            continue
        else:
            pbar.update(1)
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

            df_Q_summary.loc[len(df_Q_summary)] = [dataset['question'][i], 
                                                    int(dataset['answer_clean'][i]),
                                                    int(df_result_majority["token"][0]),
                                                    [int(i) for i in df_result_majority["token"]],
                                                    list(df_result_majority["prob"]),
                                                    entropy(list(df_result_majority["prob"])),
                                                    np.mean(index_distence_list),
                                                    i]
    df_Q_summary["correct"] = list(df_Q_summary["answer_clean"] == df_Q_summary["test_answer"])
    print("Accuracy: " + str(np.round(np.sum(df_Q_summary["answer_clean"] == df_Q_summary["test_answer"])/len(df_Q_summary)*100,2)) + "%")  

    return df_Q_summary

import string
def remove_punctuation(input_string):
    # Make a translation table that maps all punctuation characters to None
    translator = str.maketrans("", "", string.punctuation)

    # Apply the translation table to the input string
    result = input_string.translate(translator)

    return result


def df_Q_summary_process(df_Q_summary):
    print(len(df_Q_summary))
    accuracy = np.round(np.sum(df_Q_summary["answer_clean"] == df_Q_summary["test_answer"])/len(df_Q_summary)*100,2)
    print("Accuracy: " + str(accuracy) + "%")

    df_Q_summary_true = df_Q_summary[df_Q_summary["answer_clean"] == df_Q_summary["test_answer"]].reset_index(drop=True)

    Entropy_True = np.mean(df_Q_summary_true["entropy"])
    print("Entropy_True: " + str(Entropy_True))
    df_Q_summary_false = df_Q_summary[df_Q_summary["answer_clean"] != df_Q_summary["test_answer"]].reset_index(drop=True)

    Entropy_False = np.mean(df_Q_summary_false["entropy"])
    print("Entropy_False: " + str(Entropy_False))

    fig, axs = plt.subplots(2, 1, figsize=(4, 3))
    # set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 
    sns.set(style="darkgrid")
    sns.histplot(data=df_Q_summary_false, x="entropy", color="red", label="false", kde=True, ax=axs[0], bins=40)

    sns.histplot(data=df_Q_summary_true, x="entropy", color="skyblue", label="correct", kde=True, ax=axs[1], bins=40)

    plt.setp(axs, xlim = (0,max(df_Q_summary_false['entropy'])))
    plt.legend() 
    plt.show() 
    return accuracy, Entropy_True, Entropy_False



def normalize_data(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = []
    
    for value in data:
        normalized_value = (value - min_val) / (max_val - min_val)
        normalized_data.append(normalized_value)
        
    return normalized_data