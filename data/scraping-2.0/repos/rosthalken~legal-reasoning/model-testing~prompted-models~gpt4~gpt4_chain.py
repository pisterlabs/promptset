import numpy as np
import openai
import os
import pandas as pd
from   pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import time
import string

interp_dir = os.getcwd()
train_test_dir = os.path.join(interp_dir, 'train_test_splits', 'train_test_splits_2')
output_path = os.path.join(interp_dir, 'chain_of_thought_results')
generated_output_path = os.path.join(output_path, 'generations')

labeled_data_path = os.path.join(interp_dir, 'labeled_data', 'final_cleaned_paragraphs.csv')
interpretation_df = pd.read_csv(labeled_data_path)

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


openai.api_key = open(os.path.join(interp_dir, 'private', 'openai_key.txt')).read().strip()
prompt_text = "Some paragraphs in court cases interpret statutes. Within interpretation, there are two types:  grand and formal. \n\nGrand interpretation represents a legal decision that views law as an open-ended and on-going enterprise for the production and improvement of decisions that make sense on their face and in light of political, social, and economic factors.  \n\nFormal interpretation is a legal decision made according to a rule, often viewing the law as a closed and mechanical system. It screens the decision-maker off from the political, social, and economic choices involved in the decision.  \n\nLet's analyze the following passage step-by-step. First, determine if it interprets a statute. Second, if it interprets a statute, determine whether the interpretation is grand or formal. The first word in your response should label the passage with \"GRAND\", \"FORMAL\", or \"NONE\" and then explain why you chose that label. \n\n"

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_descriptive_interp(text, prompt_text):
  full_prompt = f"{prompt_text} ### \n\n Text: {text} \n\n ###"
  response = openai.ChatCompletion.create(
      model='gpt-4',
      max_tokens=2,
      messages = [{'role': 'user', 'content': full_prompt}]
  )
  return response['choices'][0]['message']['content'].strip().lower()

interpretation_df = interpretation_df[interpretation_df['class'].notna()]

macro_f1_l = []
macro_precision_l = []
macro_recall_l = []

weighted_f1_l = []
weighted_precision_l = []
weighted_recall_l = []

grand_f1_l = []
grand_precision_l = []
grand_recall_l = []

formal_f1_l = []
formal_precision_l = []
formal_recall_l = []

none_f1_l = []
none_precision_l = []
none_recall_l = []

full_df = pd.DataFrame()

for split in range(0, 5): # limit to first five splits
  start_time = time.time()

  split_id_file = os.path.join(train_test_dir, f'split_{split}')

  with open(split_id_file, 'r') as file:
      train_ids = file.read().split("\n")

  interpretation_train_df = interpretation_df[interpretation_df["section_id"].isin(train_ids)]
  interpretation_test_df = interpretation_df[~interpretation_df["section_id"].isin(train_ids)]


  X_test = interpretation_test_df["paragraph"].to_list()
  y_test = interpretation_test_df["class"].to_list()

  total = len(X_test)

  predicted_labels = []
  for i, text in enumerate(X_test):
    prediction = get_descriptive_interp(text, prompt_text).upper()
    predicted_labels.append(prediction)

    if i % 50 == 0:
      precent = round((i/total)*100, 2)
      print(f"{precent}% through processing.")

  with open(os.path.join(generated_output_path, f'predictions_{split}.txt'), 'w') as file:
     for label in predicted_labels:
        file.write(f"{label}\n")

  end_time = time.time()

  total_minutes = round((end_time - start_time) / 60, 2)
  print(f"Total time: {total_minutes} minutes.")

for split in range(0, 5):

  split_id_file = os.path.join(train_test_dir, f'split_{split}')

  with open(split_id_file, 'r') as file:
      train_ids = file.read().split("\n")

  interpretation_train_df = interpretation_df[interpretation_df["section_id"].isin(train_ids)]
  interpretation_test_df = interpretation_df[~interpretation_df["section_id"].isin(train_ids)]


  X_test = interpretation_test_df["paragraph"].to_list()
  y_test = interpretation_test_df["class"].to_list()

  with open(os.path.join(generated_output_path, f'predictions_{split}.txt'), 'r') as file:
    predicted_labels = [line.rstrip() for line in file]

  predicted_labels = [label.translate(str.maketrans('', '', string.punctuation)) for label in predicted_labels]

  class_report = classification_report(y_test, predicted_labels, output_dict=True)

  sample_dict = {
      "model": "class_examples",
      "split": split,

      "macro_f1": round(class_report["macro avg"]["f1-score"], 3),
      "macro_precision": round(class_report["macro avg"]["precision"], 3),
      "macro_recall": round(class_report["macro avg"]["recall"], 3),

      "weighted_f1": round(class_report["weighted avg"]["f1-score"], 3),
      "weighted_precision": round(class_report["weighted avg"]["precision"], 3),
      "weighted_recall": round(class_report["weighted avg"]["recall"], 3),

      "grand_f1": round(class_report["GRAND"]["f1-score"], 3),
      "grand_precision": round(class_report["GRAND"]["precision"], 3),
      "grand_recall": round(class_report["GRAND"]["recall"], 3),

      "formal_f1": round(class_report["FORMAL"]["f1-score"], 3),
      "formal_precision": round(class_report["FORMAL"]["precision"], 3),
      "formal_recall": round(class_report["FORMAL"]["recall"], 3),

      "none_f1": round(class_report["NONE"]["f1-score"], 3),
      "none_precision": round(class_report["NONE"]["precision"], 3),
      "none_recall": round(class_report["NONE"]["recall"], 3),

  }

  new_row = pd.DataFrame(sample_dict, index = [0])
  full_df = pd.concat([full_df, new_row])

  macro_f1_l.append(class_report["macro avg"]["f1-score"])
  macro_precision_l.append(class_report["macro avg"]["precision"])
  macro_recall_l.append(class_report["macro avg"]["recall"])

  weighted_f1_l.append(class_report["weighted avg"]["f1-score"])
  weighted_precision_l.append(class_report["weighted avg"]["precision"])
  weighted_recall_l.append(class_report["weighted avg"]["recall"])

  grand_f1_l.append(class_report["GRAND"]["f1-score"])
  grand_precision_l.append(class_report["GRAND"]["precision"])
  grand_recall_l.append(class_report["GRAND"]["recall"])

  formal_f1_l.append(class_report["FORMAL"]["f1-score"])
  formal_precision_l.append(class_report["FORMAL"]["precision"])
  formal_recall_l.append(class_report["FORMAL"]["recall"])

  none_f1_l.append(class_report["NONE"]["f1-score"])
  none_precision_l.append(class_report["NONE"]["precision"])
  none_recall_l.append(class_report["NONE"]["recall"])

macro_f1 = sum(macro_f1_l) / len(macro_f1_l)
macro_precision = sum(macro_precision_l) / len(macro_precision_l)
macro_recall = sum(macro_recall_l) / len(macro_recall_l)

weighted_f1 = sum(weighted_f1_l) / len(weighted_f1_l)
weighted_precision = sum(weighted_precision_l) / len(weighted_precision_l)
weighted_recall = sum(weighted_recall_l) / len(weighted_recall_l)

grand_f1 = sum(grand_f1_l) / len(grand_f1_l)
grand_precision = sum(grand_precision_l) / len(grand_precision_l)
grand_recall = sum(grand_recall_l) / len(grand_recall_l)

formal_f1 = sum(formal_f1_l) / len(formal_f1_l)
formal_precision = sum(formal_precision_l) / len(formal_precision_l)
formal_recall = sum(formal_recall_l) / len(formal_recall_l)

none_f1 = sum(none_f1_l) / len(none_f1_l)
none_precision = sum(none_precision_l) / len(none_precision_l)
none_recall = sum(none_recall_l) / len(none_recall_l)

model_dict = {
    "model": "descriptive_generative",
    "split": "averages",

    "macro_f1": round(macro_f1, 3),
    "macro_precision": round(macro_precision, 3),
    "macro_recall": round(macro_recall, 3),

    "weighted_f1": round(weighted_f1, 3),
    "weighted_precision": round(weighted_precision, 3),
    "weighted_recall": round(weighted_recall, 3),

    "grand_f1": round(grand_f1, 3),
    "grand_precision": round(grand_precision, 3),
    "grand_recall": round(grand_recall, 3),

    "formal_f1": round(formal_f1, 3),
    "formal_precision": round(formal_precision, 3),
    "formal_recall": round(formal_recall, 3),

    "none_f1": round(none_f1, 3),
    "none_precision": round(none_precision, 3),
    "none_recall": round(none_recall, 3),
}

new_row = pd.DataFrame(model_dict, index = [0])
full_df = pd.concat([full_df, new_row])

full_df.to_csv(os.path.join(output_path, 'gpt_chain.csv'))

