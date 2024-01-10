import sys

sys.path.append("../..")
from lib import model_helper, dataset_generation, automated_evaluation, chat_helper
from anthropic import Anthropic
from lib.automated_evaluation import caesar_decrypt
from transformers import AutoTokenizer
import pickle
import pandas as pd
from datagen_functions import create_paired_statements, build_HOCUS_dataset
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
import sys
import openai
import datagen_functions
import logging

# Set up logging
logging.basicConfig(
    filename="output_text.txt", level=logging.INFO, format="%(message)s"
)
import json
keys_file_path = "/root/keys"
with open(keys_file_path, 'r') as keys_file:
    keys = json.load(keys_file)

openai_api_key = keys["OPENAI_API_KEY"]


data_path = "../Datasets/HOCUS/"
HOCUS_seeds = json.load(open(data_path + "HOCUS_seed.json", "r"))
HOCUS_subtopics = json.load(open(data_path + "HOCUS_subtopics.json", "r"))
model = "gpt-4"


print("Script started")

try:
    dataset = build_HOCUS_dataset(
        HOCUS_seeds,
        HOCUS_subtopics,
        data_path,
        model,
        openai_api_key,
        N_new_pairs=25,
        logging_enabled=True,
    )
except Exception as e:
    print(f"Error: {e}")

print("Script finished")
