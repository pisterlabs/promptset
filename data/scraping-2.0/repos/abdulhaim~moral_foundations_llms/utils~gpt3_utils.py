import sys
import os
import re
import time

parent_directory =  os.path.dirname(os.getcwd())
sys.path.append(parent_directory)

import numpy as np 
import openai
from utils.questionnaire_utils import * 
NUM_SEEDS = 10 

def logprob_to_prob(logprob):
    return np.exp(logprob)

def prob_for_label(label, logprobs):
    """
    Returns the predicted probability for the given label as
    a number between 0.0 and 1.0.
    """
    # Initialize probability for this label to zero.
    prob = 0.0
    # Look at the first entry in logprobs. This represents the
    # probabilities for the very next token.
    next_logprobs = logprobs

    for s, logprob in next_logprobs.items():
        # We want labels to be considered case-insensitive. In
        # other words:
        #
        #     prob_for_label("vegetable") =
        #         prob("vegetable") + prob("Vegetable")
        #
        s = s.lower().strip()

        if label.lower() == s:
            # If the prediction matches one of the labels, add
            # the probability to the total probability for that
            # label.
            prob += logprob_to_prob(logprob)
        # elif label.lower().startswith(s):
        #     # If the prediction is a prefix of one of the labels, we
        #     # need to recur. Multiply the probability of the prefix
        #     # by the probability of the remaining part of the label.
        #     # In other words:
        #     #
        #     #     prob_for_label("vegetable") =
        #     #         prob("vege") * prob("table")
        #     #
        #     rest_of_label = label[len(s) :]
        #     remaining_logprobs = logprobs[1:]
        #     prob += logprob * prob_for_label(
        #         rest_of_label,
        #         remaining_logprobs,
        #     )
    return prob


def parse_response(response, labels):
    response = response.choices[0].text.strip()[0]
    response = labels[response] if response in labels else 6
    return response 

def compute_gpt3_generative_response(engine, prompt, labels):
    seed_answers = []
    for seed in range(NUM_SEEDS):
        response = "NA"
        while(response==6 or response=="NA"):
            response = openai.Completion.create(
                       model=engine,
                       prompt=prompt,
                       temperature=0.0,
                       max_tokens=64,
                       top_p=1.0,
                       frequency_penalty=0.0,
                       presence_penalty=0.0,
                       stop=["\"\"\""])
            response = parse_response(response, labels)
        seed_answers.append(response)
    return seed_answers


def compute_gpt3_scoring_response(engine, prompt, labels):
    response = "NA"
    while(response==6 or response=="NA"):
        response = openai.Completion.create(
                   model=engine,
                   prompt=prompt,
                   temperature=0.0,
                   max_tokens=64,
                   top_p=1.0,
                   logprobs=5,
                   frequency_penalty=0.0,
                   presence_penalty=0.0,
                   stop=["\"\"\""])
    all_prob_labels = {}
    probs = response.choices[0].logprobs.top_logprobs
    for prob_index in range(len(probs)):
        prob = probs[prob_index]
        for label, index in labels.items():
            prob_value = prob_for_label(label, prob)
            all_prob_labels[label] = prob_value
        summation = 0
        for i in all_prob_labels:
            summation+=all_prob_labels[i]
        if summation < 0.1 and prob_index!=len(probs)-1:
            all_prob_labels = {}
            continue 
        else:
            break
    answers = random.choices(list(all_prob_labels.keys()), weights=list(all_prob_labels.values()), k=num_seeds)
    answers = list(map(lambda x: labels[x], sample_values))
    return answers

def compute_gpt3_response(engine, prompt, labels, generative):
    if generative:
        return compute_gpt3_generative_response(engine, prompt, labels)
    else:
        return compute_gpt3_scoring_response(engine, prompt, labels)
    time.sleep(10) 