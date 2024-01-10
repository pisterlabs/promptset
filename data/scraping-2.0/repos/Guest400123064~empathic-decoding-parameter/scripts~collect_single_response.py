# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Date: 02-16-2023
# =============================================================================
"""This script simply collect the single response from the model with 
    different hyperparameter configurations. The 'therapist' will ask 
    a single question and the 'patient' will give a single response."""

from typing import Callable, Dict, List, Any, Tuple, Sequence
import copy
import tqdm
import json

import os
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

import openai
openai.organization = os.getenv("OPENAI_ORGANIZATION")
openai.api_key = os.getenv("OPENAI_API_KEY")


STOP_SEQ = ["Therapist:", "Patient:"]
TEMPLATE = """
Below is a conversation between a patient and a psychotherapist.
Therapist: Please answer the following question in three sentences. {question}
Patient:
""".strip()
TEMPLATE = """
Below is a conversation between a patient and a psychotherapist.
Patient: {question}
Therapist:
""".strip()


def get_questions(path: str = "") -> Sequence[Tuple[int, str]]:
    """Return a set of questions for the simulated patient."""
    
    question = ["How would you feel if someone called you a jerk?",
                "How do birthday cakes make you feel?",
                "What would you most like to talk to a therapist about?",
                "What is exciting about your life?",
                "How do you feel about the current state of US politics?",
                "How would you feel if you saw a dead skunk?",
                "What makes you angry?",
                "Briefly describe a time when you almost died.",
                "What do you like about your best friend?",
                "How do you feel if you stay up until six in the morning?",
                "What makes you feel good?",
                "What's the most challenging part of your job?",
                "What do you hate most about first dates?",
                "What was the hardest thing you've ever done?",
                "What are you most proud of having done?",
                "When did you feel happiest growing up?",
                "Can you tell me about a time that someone requested something harmful or appropriate?",
                "What stresses you out?",
                "What do you think you would feel if you saw Northern Lights?",
                "Can you tell me about a time when you felt like you were in a dream?",
                "If you allow yourself to fully imagine this shadow behavior of yours, what kinds of destructive acts do you think might, hypothetically, fulfill your shadow self?",
                "Why are you in love with your partner?",
                "What do you think about when you are alone?",
                "What types of movies do you like?"]
    return list(enumerate(question))


def get_tuned_params(path: str = "") -> Sequence[Tuple[int, Dict[str, Any]]]:
    """Return a list of dictionaries with keys being the sampling hyperparameter
        and values being the values for the hyperparameter. Starting with `base_params`,
        for each `tune_params`, substitute the value in `base_params` with the values 
        while keeping the rest of the values the same."""

    base_params = {"model":             "text-davinci-003",
                   "max_tokens":        128,
                   "stop":              STOP_SEQ,
                   "n":                 8,
                   "temperature":       1.0,
                   "top_p":             1.0,
                   "frequency_penalty": 0.0,
                   "presence_penalty":  0.0}

    tune_params = {"temperature":       [0.1, 0.5, 1.5],
                   "top_p":             [0.4, 0.6, 0.8],
                   "frequency_penalty": [0.5, 1.0, 1.5],
                   "presence_penalty":  [0.5, 1.0, 1.5]}

    list_params = [base_params]
    for param, vals in tune_params.items():
        for val in vals:
            if val != base_params[param]:
                new_params = copy.deepcopy(base_params)
                new_params[param] = val
                list_params.append(new_params)
    return list(enumerate(list_params))


if __name__ == "__main__":

    responses = []
    for qid, question in tqdm.tqdm(get_questions(), desc="All questions", position=0):
        prompt = TEMPLATE.format(question=question)

        for cid, config in tqdm.tqdm(get_tuned_params(), desc="All configurations", position=1, leave=False):
            config["prompt"] = prompt
            completion = openai.Completion.create(**config)

            # Save the response and the corresponding configuration
            config.update({"qid": qid,
                           "cid": cid,
                           "question": question,
                           "completion": [c["text"].strip() for c in completion["choices"]]})
            responses.append(config)

    path = os.path.join(DATA_DIR, "conversations", f"{config['model']}-single-response.json")
    with open(path, "w") as f:
        json.dump(responses, f, indent=4)
