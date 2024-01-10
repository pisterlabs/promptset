import sys
import pandas as pd
sys.path.append(r"C:\Users\silvh\OneDrive\lighthouse\custom_python")
sys.path.append(r"C:\Users\silvh\OneDrive\lighthouse\portfolio-projects\online-PT-social-media-NLP\src")
sys.path.append(r"C:\Users\silvh\OneDrive\lighthouse\Ginkgo coding\content-summarization\src")
from summarization import Chatbot, reply
from summary_chain import *
from article_processing import create_text_dict
from silvhua import *
from datetime import datetime
import openai
import os
import re
from itertools import product
import time

save_outputs = True
folder_path = '../text/2023-04-17'
prompts_dict = dict()
experiment_num = 1
relevance_prompts_dict = dict()

encoding='ISO-8859-1'

all_files = []

prep_step = [
    # "Take the key points to",
    "Take the key points and numerical descriptors to",
    # ""
]

task_part1 = [
    # "Summarize the article in under 300 characters",
    "Summarize for a LinkedIn post",
    # "Summarize for a tweet",
    # "Summarize in an engaging way",
    "Describe the interesting points to your coworker at the water cooler"
    # "Summarize the article for a Tiktok post"
]
audience = [
    # "lay audience",
    # "",
    "seniors",
    "people who enjoy sports"
]
task_part2 = [
    "",
    "Use terms a 12-year-old can understand.",
    # "Assume your audience has no science background."
    # "Include the most interesting findings.",
    # "Include the key take-aways for the reader.",
    # "Include the implications of the article."
]

task_part3 =[
    "Add 1-2 sentences to make this relevant for"
    # "Add 1-2 sentences to make this relevant for older adults."
    # "Once you are done, add 1-2 sentences to make this relevant for older adults.",
    ""
]

# prompts_df = pd.DataFrame(product(prep_step, task_part1, task_part2, task_part3, audience), 
#     columns=['prep_step', 'task part 1', 'task part 2', 'task part 3', 'audience'])

prompts_df = pd.DataFrame(product(prep_step, task_part1), 
    columns=['prep_step', 'task part 1'])

relevance_prompts_df = pd.DataFrame(product(task_part3, audience), 
    columns=['task part 3', 'audience'])

prompts_df['prompt'] = prompts_df.apply(
    lambda row: f"{row['prep_step']} {row['task part 1']}.", 
    axis=1)
# prompts_df['simplify'] = prompts_df.apply(
#     lambda row: f" {row['task part 2'] if row['task part 2'] else ''}", 
#     axis=1)
relevance_prompts_df['relevance'] = relevance_prompts_df.apply(
    lambda row: f" {row['task part 3']} {row['audience']} " if row['audience'] else '', 
    axis=1) 

for filename in os.listdir(folder_path):
    with open(os.path.join(folder_path, filename), 'r', encoding=encoding) as f:
        all_files.append(f.read())

text_dict = create_text_dict(all_files)

iteration_id = experiment_num
n_choices = 5
qna_dict = dict()
qna_chain_dict = dict()
chatbot_dict = dict()
simplify_prompts = task_part2
summary_iteration_id = iteration_id
pause_per_request=20
relevance_prompts = relevance_prompts_df

qna_dict, chaining_dict = batch_summarize_chain(text_dict, prompts_df, qna_dict, chatbot_dict, 
    n_choices=n_choices, pause_per_request=1,
    iteration_id=iteration_id)

time.sleep(20)

qna_dict = prompt_chaining_dict(simplify_prompts, qna_dict, chaining_dict[summary_iteration_id], iteration_id,
    n_choices=1, pause_per_request=pause_per_request,
    simplify_iteration=1, summary_iteration_id=summary_iteration_id, save_outputs=True
    )
qna_dict = prompt_chaining_dict(relevance_prompts, qna_dict, chaining_dict[summary_iteration_id], iteration_id,
    prompt_column='relevance', n_choices=n_choices, pause_per_request=pause_per_request,
    simplify_iteration=1, summary_iteration_id=summary_iteration_id, save_outputs=save_outputs
    )
# print(os.getenv('api_openai'))