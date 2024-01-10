from helper import *
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, OPTForCausalLM, GPT2Tokenizer
from sklearn.utils import shuffle
from torch import optim
from tqdm import tqdm
import time
import json
import csv
import random
import numpy as np
import pandas as pd
from helper import get_model_and_tokenizer


# for generating set prompts from ground-truth aspects
def few_shot_prompt(q, add_answer=False):
    text = ""
    for key in list(q['correctness_explanation'].keys()):
        text += "input: " + str(key) + ' \n'
        text += "output: "
        if add_answer:
            text += q['options'][q['answer']] + " \n"

    return text




def get_fewshot_NLL_score(model,tokenizer,condition,text,filler=' so I recommend ',normalize=False):
    text = condition + filler  + text
    encodings = tokenizer(text, return_tensors="pt")
    condition = tokenizer(condition, return_tensors="pt")
    stride = condition.input_ids.size(1)

    nlls = []

    begin_loc = 0
    end_loc = stride
    trg_len = encodings.input_ids.size(1) - stride
    input_ids = encodings.input_ids.to('cuda')
    target_ids = input_ids.clone()

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs[0]

    if normalize:
        with torch.no_grad():
            c_input_ids = condition.input_ids.to('cuda')
            outputs = model(c_input_ids, labels=c_input_ids)
            c_neg_log_likelihood = outputs[0]
        return (-1 * neg_log_likelihood) - (-1 * c_neg_log_likelihood)
    else:
        return -1 * neg_log_likelihood



def aspect_FS_pred(train_data,test_dataset, model_name, agg_fcn, normalize=True,prompt_size=5):
    model_class, tokenizer_class = get_model_and_tokenizer(model_name)
    model = model_class.from_pretrained(model_name).to('cuda')
    tokenizer = tokenizer_class.from_pretrained(model_name)


    output_message = ""
    predictions = []
    # loop through each query
    for sample in test_dataset:
        # count += 1
        # if count % 50 == 0:
        #     print('--> ',count)

        # for key in sample['query_type']:
        #     if sample['query_type'][key] == 1:
        #         type_count[key] += 1
        prompt = ''
        # prompt_size = 5
        # generate prompt sample
        for index in range(prompt_size):
            p = few_shot_prompt(train_data[index],True)
            prompt += p
        # output_message += str(count) + ' Query: ' + sample["query"] + ' \n'

        q_text = sample["query"]
        aspects = sample["correctness_explanation"].keys()
        options_list = [val for val in sample["options"].values()]

        all_scores = []
        for a in aspects: 
            p ="input: " + a + "\n"
            q_text = prompt + p
            scores = []

            for key in sample["options"]:
                score = get_fewshot_NLL_score(model, tokenizer, q_text, sample["options"][key], normalize=normalize, filler='')
                assert not torch.isnan(score), 'score is nan'
                scores.append(float(score))

                # if key == sample["answer"]:
                #     output_message += 'Answer: ' + str(score) + ' ' + sample["options"][key] + ' \n'
                # else:
                #     output_message += str(score) + ' ' + sample["options"][key] + ' \n'

            all_scores.append(scores)

        agg_scores = aggregate(all_scores, agg_fcn)
        agg_scores, options_list = shuffle(agg_scores, options_list, random_state=0)
        args = np.argsort(agg_scores)
        predicted_id = options_list[args[-1]]

        
        predictions.append(predicted_id)
    return predictions