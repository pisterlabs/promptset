import sys

sys.path.append("../..")
from lib import automated_evaluation, chat_helper
from anthropic import Anthropic
from lib.automated_evaluation import caesar_decrypt
from lib.hallucination_lib import get_letter_probabilities_batched
from transformers import AutoTokenizer
import pickle
import pandas as pd
from tqdm import tqdm
import torch
import json

keys_file_path = "/root/keys"
with open(keys_file_path, "r") as keys_file:
    keys = json.load(keys_file)

token = keys["TOKEN_NAME"]
import os
import re
import json
import numpy as np

system_prompt = (
    "You are a helpful, honest and concise assistant."  # Needed for chat model
)
print("Loading model...")
model = chat_helper.Llama7BChatHelper(
    token, system_prompt, master_device=0, threshold=0.1
)
model.tokenizer.pad_token = model.tokenizer.eos_token


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


def generate_answers(questions, model, N=1, truth_question=True, fiction_question=True):
    questions = (
        questions.copy()
    )  # Create a copy of the DataFrame to ensure you're not modifying a view
    for i in range(N):
        if truth_question:
            answer_name = "truth_answer_" + str(i)
            if answer_name not in questions.columns:
                q_list = questions["truth_question"].tolist()
                a_list = model.generate_text_batched(
                    q_list, max_length=200, max_tokens_per_batch=2000
                )
                a_list = [a.split("[/INST]")[-1].strip() for a in a_list]

                questions.loc[:, answer_name] = a_list

        if fiction_question:
            answer_name = "fiction_answer_" + str(i)
            if answer_name not in questions.columns:
                q_list = questions["fiction_question"].tolist()
                a_list = model.generate_text_batched(
                    q_list, max_length=200, max_tokens_per_batch=2000
                )
                a_list = [a.split("[/INST]")[-1].strip() for a in a_list]

                questions.loc[:, answer_name] = a_list

    return questions


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


def rate_all_answers_wrapper_function(questions, N=20):
    # Splitting the main dataframe into smaller dataframes of size N
    list_of_dfs = [questions.iloc[i : i + N] for i in range(0, len(questions), N)]

    # Process each smaller dataframe
    processed_dfs = []
    for small_df in list_of_dfs:
        rate_all_answers(small_df)
        processed_dfs.append(small_df)

    # Combine the processed dataframes to get the final dataframe
    final_df = pd.concat(processed_dfs, axis=0, ignore_index=True)

    return final_df


def rate_all_answers(questions):
    for key in questions.keys():
        if "truth_answer" in key:
            i = int(re.match(r"truth_answer_(\d+)$", key).group(1))
            answer_name = "truth_rating_" + str(i)
            if answer_name not in questions.keys():
                rate_df = pd.DataFrame(
                    {
                        "questions": questions["truth_question"].tolist(),
                        "answers": questions[key].tolist(),
                        "solutions": questions["truth"].tolist(),
                    }
                )
                automated_evaluation.rate_answers_to_true_questions(rate_df)
                questions[answer_name] = rate_df["rating"].tolist()
                questions["truth_rating_text_" + str(i)] = rate_df[
                    "rating_text"
                ].tolist()
        if "fiction_answer" in key:
            i = int(re.match(r"fiction_answer_(\d+)$", key).group(1))
            answer_name = "fiction_rating_" + str(i)
            if answer_name not in questions.keys():
                rate_df = pd.DataFrame(
                    {
                        "question": questions["fiction_question"].tolist(),
                        "answers": questions[key].tolist(),
                        "fiction": questions["fiction"].tolist(),
                        "truth": questions["truth"].tolist(),
                    }
                )
                automated_evaluation.classify_affirmation_answer(rate_df)
                questions[answer_name] = rate_df["rating"].tolist()
                questions["fiction_rating_text_" + str(i)] = rate_df[
                    "rating_text"
                ].tolist()


def format_text(text):
    # Replace Human speech with [INST] tags
    text = re.sub(r"Human: (.*?)(?=\n|$)", r"[INST] \1 [/INST]", text)
    # Remove AI: prefix
    text = re.sub(r"AI: ", "", text)

    # Remove first [INST] and last [/INST] tags
    text = text.replace("[INST]", "", 1)  # Remove first occurrence only
    sub = "[/INST]"
    text = (
        text[: text.rfind(sub)] + text[text.rfind(sub) + len(sub) :]
        if sub in text
        else text
    )

    return text


question_path = "../steering_vectors/"
question_types = [
    "direct_questions",
    "questioning_assuming_statement",
    "conversation",
    "alluding_questions",
]
for question_type in question_types:
    questions = pd.read_csv(f"{question_path}{question_type}/_test.csv")
    if question_type == "conversation":
        questions["fiction_question"] = questions["fiction_question"].apply(format_text)
        questions["truth_question"] = questions["truth_question"].apply(format_text)
    questions = questions.drop(questions.index[::20]).reset_index(drop=True)
    questions_comparison = questions[::4].reset_index(drop=True)
    print(len(questions_comparison))
    path = f"./steered_completions/{question_type}/"
    if not os.path.exists(path):
        os.makedirs(path)
    layer = 15
    coeff_list = [-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10]
    steering_data_fiction = torch.load(
        f"{question_path}{question_type}/fiction/all_diffs_layer_{layer}.pt"
    )

    steering_vector_fiction = average_random_lines(steering_data_fiction)
    steering_vector_fiction = steering_vector_fiction / np.linalg.norm(
        steering_vector_fiction
    )

    steering_data_mix = torch.load(
        f"{question_path}{question_type}/mixed/all_diffs_layer_{layer}.pt"
    )

    steering_vector_mix = average_random_lines(steering_data_mix)
    steering_vector_mix = steering_vector_mix / np.linalg.norm(steering_vector_mix)

    steering_data_truth = torch.load(
        f"{question_path}{question_type}/truth/all_diffs_layer_{layer}.pt"
    )

    steering_vector_truth = average_random_lines(steering_data_truth)
    steering_vector_truth = steering_vector_truth / np.linalg.norm(
        steering_vector_truth
    )

    steering_vector_added = steering_vector_fiction + steering_vector_truth

    # steering_data_truth = torch.load(f'{question_path}{question_type}/truth/all_diffs_layer_{layer}.pt')
    # steering_vector_truth = average_random_lines(steering_data_truth)
    # steering_vector_combined = (steering_vector_fiction + steering_vector_truth)/2

    for coeff in coeff_list:
        if not os.path.exists(f"{path}fiction_steered_{coeff}.csv"):
            print(
                f"generating fiction steered completions for {question_type} with coeff {coeff}"
            )
            model.reset_all()
            model.set_add_activations(layer, steering_vector_fiction * coeff)
            fiction_steered_completions = generate_answers(questions_comparison, model)
            fiction_steered_completions.to_csv(
                f"{path}fiction_steered_{coeff}.csv", index=False
            )
        else:
            print(
                f"skipping fiction steered completions for {question_type} with coeff {coeff}"
            )
        if not os.path.exists(f"{path}mix_steered_{coeff}.csv"):
            print(
                f"generating mix steered completions for {question_type} with coeff {coeff}"
            )
            model.reset_all()
            model.set_add_activations(layer, steering_vector_mix * coeff)
            mix_steered_completions = generate_answers(questions_comparison, model)
            mix_steered_completions.to_csv(
                f"{path}mix_steered_{coeff}.csv", index=False
            )
        else:
            print(
                f"skipping mix steered completions for {question_type} with coeff {coeff}"
            )
        if not os.path.exists(f"{path}added_steered_{coeff}.csv"):
            print(
                f"generating truth steered completions for {question_type} with coeff {coeff}"
            )
            model.reset_all()
            model.set_add_activations(layer, steering_vector_added * coeff)
            truth_steered_completions = generate_answers(questions_comparison, model)
            truth_steered_completions.to_csv(
                f"{path}added_steered_{coeff}.csv", index=False
            )
