import sys

sys.path.append("../..")
from lib import model_helper, dataset_generation, automated_evaluation, chat_helper
from anthropic import Anthropic
from lib.automated_evaluation import caesar_decrypt
from transformers import AutoTokenizer
import pickle
import pandas as pd
from datagen_functions import (
    direct_questioning,
    questioning_assuming_statement,
    alluding_questioning,
)
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
import openai

import json
keys_file_path = "/root/keys"
with open(keys_file_path, 'r') as keys_file:
    keys = json.load(keys_file)

data_path = "../Datasets/HOCUS/"
HOCUS_seeds = json.load(open(data_path + "HOCUS_seed.json", "r"))


def add_questions_to_topic(data_frame, question_function):
    # Split the data into truth and fiction DataFrames
    truth_df = pd.DataFrame({"statement": data_frame["truth"].dropna().tolist()})
    fiction_df = pd.DataFrame({"statement": data_frame["fiction"].dropna().tolist()})

    # Generate questions for truth and fiction statements
    question_function(truth_df)
    question_function(fiction_df)

    # Update the original DataFrame with the generated questions
    # Update the original DataFrame with the generated questions
    data_frame["truth_question"] = truth_df["question"].values
    data_frame["fiction_question"] = fiction_df["question"].values


from tqdm import tqdm

# Load existing data if it exists
if os.path.exists(data_path + "/questions/" + "statements.csv"):
    complete_df = pd.read_csv(data_path + "/questions/" + "statements.csv")
    processed_subtopics = set(complete_df["sub_topic"].unique())
else:
    complete_df = pd.DataFrame()
    processed_subtopics = set()

topic_folders = os.listdir(data_path + "/statements")
dfs = []  # List to store DataFrames for each topic

# Calculate total number of subtopics for the progress bar
total_subtopics = sum(
    [len(os.listdir(data_path + "/statements/" + topic)) for topic in topic_folders]
)

with tqdm(total=total_subtopics, desc="Subtopics") as pbar:
    for topic in topic_folders:
        files = os.listdir(data_path + "/statements/" + topic)
        sub_topics = [file.replace(".json", "") for file in files]

        for sub_topic in sub_topics:
            if sub_topic in processed_subtopics:
                print("Skipping " + sub_topic)
                pbar.update(1)  # Update progress bar even if skipping
                continue  # Skip this subtopic if it's already processed
            print("Processing " + sub_topic)
            all_statements = []
            statements_data = json.load(
                open(
                    data_path + "/statements/" + topic + "/" + sub_topic + ".json", "r"
                )
            )

            # Add topic and subtopic information to each statement
            for statement in statements_data:
                statement["topic"] = topic
                statement["sub_topic"] = sub_topic
                all_statements.append(statement)

            # Convert the list of dictionaries for this topic to a DataFrame and add to the dfs list
            df = pd.DataFrame(all_statements)
            print("Generating questions for " + sub_topic)
            add_questions_to_topic(df, direct_questioning)
            print("Finished generating questions for " + sub_topic)
            dfs.append(df)

            # Save progress
            complete_df = pd.concat([complete_df, df], ignore_index=True)
            complete_df.to_csv(
                data_path + "/questions/" + "statements.csv", index=False
            )

            pbar.update(1)  # Update progress bar after processing each subtopic

# If you want to save the complete DataFrame to a CSV file (this might be redundant if you're saving after each subtopic):
# complete_df.to_csv(data_path + "/questions/" + "statements.csv", index=False)
