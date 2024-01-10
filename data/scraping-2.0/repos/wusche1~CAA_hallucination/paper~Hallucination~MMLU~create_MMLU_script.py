import sys

sys.path.append("../..")
from lib import automated_evaluation, chat_helper
from anthropic import Anthropic
from lib.automated_evaluation import caesar_decrypt
from lib.hallucination_lib import get_letter_probabilities_batched
from transformers import AutoTokenizer
import pickle
import random
import pandas as pd
import datasets
import torch.nn.functional as F
from tqdm import tqdm
import torch

token = os.getenv("TOKEN_NAME")
import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np

system_prompt = (
    "You are a helpful, honest and concise assistant."  # Needed for chat model
)

model = chat_helper.Llama7BChatHelper(
    token, system_prompt, master_device=0, threshold=0.1
)
model.tokenizer.pad_token = model.tokenizer.eos_token


def get_mmlu_questions(number, subsets):
    result = []
    for subset in subsets:
        try:
            dataset = datasets.load_dataset("lukaemon/mmlu", subset)["test"]
        except Exception as e:
            print(f"Failed to load dataset for subset {subset}. Error: {e}")
            continue
        # Randomly sample 'number' items from the data
        samples = random.sample(list(dataset), min(number, len(dataset)))
        for sample in samples:
            # Dynamically build the question with choices
            question_parts = [sample["input"]]
            for choice_letter in ["A", "B", "C", "D"]:
                if choice_letter in sample:
                    question_parts.append(f"{choice_letter}: {sample[choice_letter]}")
                else:
                    print(f"letter {choice_letter} not in sample {sample}")
            question_parts.append("I choose ")
            question = "\n".join(question_parts)
            answer = sample.get("target", "Unknown")
            result.append({"question": question, "answer": answer})
    return result


def find_token_ids_for_letter(tokenizer, letter: str):
    letter = letter.lower()  # Convert the letter to lowercase
    matching_ids = []
    # Iterate over tokens and their ids
    for token, token_id in tokenizer.get_vocab().items():
        if token.lower().replace(" ", "") == letter:
            matching_ids.append(token_id)
    return matching_ids


def get_mmlu_probabilities(model, question):
    assert question["answer"] in [
        "A",
        "B",
        "C",
        "D",
    ], "The letter for the answer is not a valid letter"
    test_prompt_tokens = chat_helper.prompt_to_tokens(
        model.tokenizer,
        system_prompt,
        question["question"],
        "I choose (",
    )
    test_prompt_tokens = test_prompt_tokens.to(model.device)
    logits = model.get_logits(test_prompt_tokens)[0, -1, :]
    probabilities = F.softmax(logits, dim=-1).to("cpu")
    ids_for_answer = find_token_ids_for_letter(model.tokenizer, question["answer"])
    answer_probability = round(sum(probabilities[ids_for_answer]).item(), 2)
    return {"question": question, "answer_probability": answer_probability}


def get_mmlu_average(model, questions):
    probabilities = []
    for question in questions:
        probabilities.append(
            get_mmlu_probabilities(model, question)["answer_probability"]
        )
    return sum(probabilities) / len(probabilities)


questions = get_mmlu_questions(
    5,
    [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ],
)


def average_random_lines(data, N=100):
    """
    This function selects 100 random rows from the input dataset and returns their average.

    Parameters:
    - data (torch.Tensor): The input dataset.

    Returns:
    - torch.Tensor: A tensor containing the average of the randomly selected 100 rows.
    """

    # Randomly select 100 indices
    indices = torch.randperm(data.size(0))[:N]

    # Select the rows corresponding to these indices
    selected_data = data[indices]

    # Return the average of the selected rows
    return torch.mean(selected_data, dim=0)


question_path = "../steering_vectors/"
question_types = [
    "direct_questions",
    "questioning_assuming_statement",
    "conversation",
    "alluding_questions",
]
steering_vectors_fiction = []
steering_vectors_mixed = []
layer = 15
coeff_list = [-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10]
for question_type in question_types:
    steering_data_fiction = torch.load(
        f"{question_path}{question_type}/fiction/all_diffs_layer_{layer}.pt"
    )
    steering_vector_fiction = average_random_lines(steering_data_fiction)
    steering_vectors_fiction.append(steering_vector_fiction)

    steering_data_mixed = torch.load(
        f"{question_path}{question_type}/mixed/all_diffs_layer_{layer}.pt"
    )
    steering_vector_mixed = average_random_lines(steering_data_mixed)
    steering_vectors_mixed.append(steering_vector_mixed)

from tqdm import tqdm

result_dict_fiction = {}
result_dict_mixed = {}

# Loop over each steering vector for fiction data, with a progress bar
for steering_vector, steering_vector_name in zip(
    steering_vectors_fiction, question_types
):
    for coeff in tqdm(coeff_list, desc=f"Processing fiction {steering_vector_name}"):
        model.reset_all()
        model.set_add_activations(layer, steering_vector * coeff)
        result = get_mmlu_average(model, questions)
        result_dict_fiction[f"{steering_vector_name}_{coeff}"] = result

# Save the results
with open("results_mmlu_fiction.pkl", "wb") as f:
    pickle.dump(result_dict_fiction, f)

"""
# Loop over each steering vector for fiction data, with a progress bar
for steering_vector, steering_vector_name in zip(
    steering_vectors_fiction, question_types
):
    for coeff in tqdm(coeff_list, desc=f"Processing fiction {steering_vector_name}"):
        model.reset_all()
        model.set_add_activations(layer, steering_vector * coeff)
        result = get_mmlu_average(model, questions)
        result_dict_fiction[f"{steering_vector_name}_{coeff}"] = result

# Save the results
with open("results_mmlu_fiction.pkl", "wb") as f:
    pickle.dump(result_dict_fiction, f)

# Loop over each steering vector for mixed data, with a progress bar
for steering_vector, steering_vector_name in zip(
    steering_vectors_mixed, question_types
):
    for coeff in tqdm(coeff_list, desc=f"Processing mixed {steering_vector_name}"):
        model.reset_all()
        model.set_add_activations(layer, steering_vector * coeff)
        result = get_mmlu_average(model, questions)
        result_dict_mixed[f"{steering_vector_name}_{coeff}"] = result
# Save the results
with open("results_mmlu_fiction.pkl", "wb") as f:
    pickle.dump(result_dict_fiction, f)

with open("results_mmlu_mixed.pkl", "wb") as f:
    pickle.dump(result_dict_mixed, f)
"""
