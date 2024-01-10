import sys

sys.path.append("../../..")
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
with open(keys_file_path, 'r') as keys_file:
    keys = json.load(keys_file)

token = keys["TOKEN_NAME"]
import os
import re
import json
import numpy as np

system_prompt = (
    "You are a helpful, honest and concise assistant."  # Needed for chat model
)
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


question_path = "../../Datasets/HOCUS/questions/"
conversation_all_questions = pd.read_csv(question_path + "conversation.csv")
conversations_steering_questions = conversation_all_questions.iloc[1::2]
steering_topics = conversations_steering_questions["topic"].tolist()

question_path = "../../steering_vectors/"
question_type = "conversation"
layer = 15
steering_data_fiction = torch.load(
    f"{question_path}{question_type}/fiction/all_diffs_layer_{layer}.pt"
)
questions = pd.read_csv(f"{question_path}{question_type}/_test.csv")

data_list = steering_data_fiction.tolist()

# Create the DataFrame
steering_df = pd.DataFrame({"topic": steering_topics, "steering_vector": data_list})

test_data_path = "../../steering_vectors/"
test_data = pd.read_csv(f"{test_data_path}{question_type}/_test.csv")
test_data["fiction_question"] = test_data["fiction_question"].apply(format_text)
test_data["truth_question"] = test_data["truth_question"].apply(format_text)

topics = np.unique(steering_topics)
# Shuffle the vector randomly
np.random.shuffle(topics)

# Split into roughly equal parts
topic_splits = np.array_split(topics, 4)
test_splits = []
steering_splits = []
for tolic_part in topic_splits:
    test_splits.append(test_data[test_data["topic"].isin(tolic_part)])
    steering_splits.append(steering_df[steering_df["topic"].isin(tolic_part)])
    print(list(tolic_part))

save_path = "./completion/"
layer = 15
coeff = -0.5
if not os.path.exists(save_path):
    os.makedirs(save_path)
for i, (topics, test_data) in enumerate(zip(topic_splits, test_splits)):
    path = f"{save_path}{i}"
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(f"{path}/topics.csv"):
        pd.DataFrame({"topics": topics}).to_csv(f"{path}/topics.csv", index=False)
    if not os.path.exists(f"{path}/unsteered.csv"):
        model.reset_all()
        unsteered_completions = generate_answers(test_data, model)
        unsteered_completions.to_csv(f"{path}/unsteered.csv", index=False)

    for j, steering_data in enumerate(steering_splits):
        steering_array = torch.tensor(steering_data["steering_vector"].tolist())
        steering_vector = average_random_lines(steering_array)
        if not os.path.exists(f"{path}/steered_{j}.csv"):
            model.reset_all()
            model.set_add_activations(layer, steering_vector * coeff)
            steered_completions = generate_answers(test_data, model)
            steered_completions.to_csv(f"{path}/steered_{j}.csv", index=False)
