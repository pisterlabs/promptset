import sys

sys.path.append("../..")
from lib import model_helper, dataset_generation, automated_evaluation, chat_helper
from anthropic import Anthropic
from lib.automated_evaluation import caesar_decrypt
from transformers import AutoTokenizer
import pickle
import pandas as pd
from tqdm import tqdm
import torch

import json
keys_file_path = "/root/keys"
with open(keys_file_path, 'r') as keys_file:
    keys = json.load(keys_file)

token = keys["TOKEN_NAME"]
import os
import numpy as np
import re
import json

system_prompt = (
    "You are a helpful, honest and concise assistant."  # Needed for chat model
)
question_path = "../Datasets/HOCUS/questions/"
direct_questions = pd.read_csv(question_path + "direct_questions.csv")
alluding_questions = pd.read_csv(question_path + "alluding_questioning.csv")
questioning_assuming_statement = pd.read_csv(
    question_path + "questioning_assuming_statement.csv"
)
conversation = pd.read_csv(question_path + "conversation.csv")

model = chat_helper.Llama7BChatHelper(
    token, system_prompt, master_device=0, threshold=0.1
)
model.tokenizer.pad_token = model.tokenizer.eos_token


def question_dataset_to_steering_dataset(question_dataset):
    truth_dataset = []
    fiction_dataset = []
    mixed_dataset = []
    for _, row in question_dataset.iterrows():
        truth_prompt = row["truth_question"]
        truth_prompt += "\n\nChoices:\n"
        rand_bool = np.random.randint(0, 2)
        if rand_bool == 0:
            truth_prompt += "\n(A): " + row["truth_accepetance"]
            truth_prompt += "\n(B): " + row["truth_rejection"]

            truth_chosen = "I choose (A)"
            truth_rejected = "I choose (B)"
        else:
            truth_prompt += "\n(A): " + row["truth_rejection"]
            truth_prompt += "\n(B): " + row["truth_accepetance"]

            truth_chosen = "I choose (B)"
            truth_rejected = "I choose (A)"

        fiction_prompt = row["fiction_question"]
        fiction_prompt += "\n\nChoices:\n"

        if rand_bool == 0:
            fiction_prompt += "\n(A): " + row["fiction_rejection"]
            fiction_prompt += "\n(B): " + row["fiction_accepetance"]

            fiction_chosen = "I choose (A)"
            fiction_rejected = "I choose (B)"

        else:
            fiction_prompt += "\n(A): " + row["fiction_accepetance"]
            fiction_prompt += "\n(B): " + row["fiction_rejection"]

            fiction_chosen = "I choose (B)"
            fiction_rejected = "I choose (A)"

        question_1 = row["truth_question"]
        question_2 = row["fiction_question"]

        chosen_1 = row["truth_accepetance"]
        chosen_2 = row["fiction_rejection"]

        rejected_1 = row["truth_rejection"]
        rejected_2 = row["fiction_accepetance"]

        if rand_bool:
            question_1, question_2 = question_2, question_1
            chosen_1, chosen_2 = chosen_2, chosen_1
            rejected_1, rejected_2 = rejected_2, rejected_1

        mixed_prompt = "Question1:" + question_1 + " Question2:" + question_2

        wrong_choices = [
            "Answer1:" + chosen_1 + "Answer2: " + rejected_2,
            "Answer1:" + rejected_1 + "Answer2: " + chosen_2,
            "Answer1:" + rejected_1 + "Answer2: " + rejected_2,
        ]
        right_choice = "Answer1:" + chosen_1 + "Answer2: " + chosen_2
        all_choices = wrong_choices + [right_choice]

        permutation = np.random.permutation(4)

        mixed_prompt += "\n\nChoices:\n"
        mixed_prompt += "\n(A): " + all_choices[permutation[0]]
        mixed_prompt += "\n(B): " + all_choices[permutation[1]]
        mixed_prompt += "\n(C): " + all_choices[permutation[2]]
        mixed_prompt += "\n(D): " + all_choices[permutation[3]]

        # Identify the correct choice
        correct_choice_index = np.where(permutation == 3)[0][0]
        correct_choice_label = ["(A)", "(B)", "(C)", "(D)"][correct_choice_index]

        # Randomly select a rejected choice
        other_choices = set(["(A)", "(B)", "(C)", "(D)"]) - set([correct_choice_label])
        rejected_choice_label = np.random.choice(list(other_choices))

        mixed_chosen = f"I choose {correct_choice_label}"
        mixed_rejected = f"I choose {rejected_choice_label}"

        truth_dataset.append(
            {"prompt": truth_prompt, "chosen": truth_chosen, "rejected": truth_rejected}
        )
        fiction_dataset.append(
            {
                "prompt": fiction_prompt,
                "chosen": fiction_chosen,
                "rejected": fiction_rejected,
            }
        )
        mixed_dataset.append(
            {"prompt": mixed_prompt, "chosen": mixed_chosen, "rejected": mixed_rejected}
        )

    return truth_dataset, fiction_dataset, mixed_dataset


datasets = [
    direct_questions,
    alluding_questions,
    questioning_assuming_statement,
    conversation,
]
dataset_path_names = [
    "direct_questions",
    "alluding_questions",
    "questioning_assuming_statement",
    "conversation",
]
for dataset, name in zip(datasets, dataset_path_names):
    test_data = dataset.iloc[::2]
    steering_data = dataset.iloc[1::2]
    path = "./" + name

    os.makedirs(path, exist_ok=True)
    test_data.to_csv(path + "/_test.csv", index=False)

    (
        steering_truth,
        steering_fiction,
        steering_mixed,
    ) = question_dataset_to_steering_dataset(steering_data)
    steering_truth_dataset = chat_helper.ComparisonDataset(
        steering_truth, system_prompt, "meta-llama/Llama-2-7b-chat-hf"
    )
    steering_fiction_dataset = chat_helper.ComparisonDataset(
        steering_fiction, system_prompt, "meta-llama/Llama-2-7b-chat-hf"
    )
    steering_mixed_dataset = chat_helper.ComparisonDataset(
        steering_mixed, system_prompt, "meta-llama/Llama-2-7b-chat-hf"
    )

    truth_path = path + "/truth"
    fiction_path = path + "/fiction"
    mixed_path = path + "/mixed"

    os.makedirs(truth_path, exist_ok=True)
    os.makedirs(fiction_path, exist_ok=True)
    os.makedirs(mixed_path, exist_ok=True)
    # only generate vectors if files are emoty
    if len(os.listdir(truth_path)) == 0:
        print("generating steering vectors for ", name)
        chat_helper.generate_and_save_steering_vectors(
            model,
            steering_truth_dataset,
            start_layer=13,
            end_layer=30,
            data_path=truth_path,
            save_all_diffs=True,
        )
    else:
        print("skipping ", name)
    if len(os.listdir(fiction_path)) == 0:
        chat_helper.generate_and_save_steering_vectors(
            model,
            steering_fiction_dataset,
            start_layer=13,
            end_layer=30,
            data_path=fiction_path,
            save_all_diffs=True,
        )
        print("generating steering vectors for ", name)
    else:
        print("skipping ", name)
    if len(os.listdir(mixed_path)) == 0:
        chat_helper.generate_and_save_steering_vectors(
            model,
            steering_mixed_dataset,
            start_layer=13,
            end_layer=30,
            data_path=mixed_path,
            save_all_diffs=True,
        )
        print("generating steering vectors for ", name)
    else:
        print("skipping ", name)
