import numpy as np
import openai
import os
import pandas as pd
from   pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import time

interp_dir = os.getcwd()
train_test_dir = os.path.join(interp_dir, 'train_test_splits', 'train_test_splits_2')
output_path = os.path.join(interp_dir, 'interpretation_results')
generated_output_path = os.path.join(output_path, 'generations')

labeled_data_path = os.path.join(interp_dir, 'labeled_data', 'final_cleaned_paragraphs.csv')
interpretation_df = pd.read_csv(labeled_data_path)

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

device_name = 'cuda'
max_length = 512

openai.api_key = open(os.path.join(interp_dir, 'private', 'openai_key.txt')).read().strip()

interpretation_prompt = "Some paragraphs in court cases interpret statutes. In this type of paragraph, there is an analysis of a statute and a claim made about its meaning. \n\nIn the following paragraph, determine if legal interpretation occurs. If yes, respond with \”interpretation\” and if not, respond with \”no interpretation\”"

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_binary_interp(text, interpretation_prompt):
  response = openai.ChatCompletion.create(
      model='gpt-4',
      max_tokens=5,
      messages = [{'role': 'user', 'content': interpretation_prompt + text}]
  )
  return response['choices'][0]['message']['content'].strip().lower()

interpretation_df = interpretation_df[interpretation_df['class'].notna()]
interpretation_df["interpretation"] = np.where(interpretation_df["class"].isin(["FORMAL", "GRAND"]), "interpretation", "no interpretation")

macro_f1_l = []
macro_precision_l = []
macro_recall_l = []

weighted_f1_l = []
weighted_precision_l = []
weighted_recall_l = []

one_f1_l = []
one_precision_l = []
one_recall_l = []

zero_f1_l = []
zero_precision_l = []
zero_recall_l = []

full_df = pd.DataFrame()

for split in range(0, 5):
  start_time = time.time()
  split_id_file = os.path.join(train_test_dir, f'split_{split}')
  with open(split_id_file, 'r') as file:
      train_ids = file.read().split("\n")

  interpretation_train_df = interpretation_df[interpretation_df["section_id"].isin(train_ids)]
  interpretation_test_df = interpretation_df[~interpretation_df["section_id"].isin(train_ids)]

  X_test = interpretation_test_df["paragraph"].to_list()
  y_test = interpretation_test_df["interpretation"].to_list()

  total = len(X_test)

  predicted_labels = []
  for i, text in enumerate(X_test):
    prediction = get_binary_interp(text, interpretation_prompt)
    predicted_labels.append(prediction)

    if i % 50 == 0:
      precent = round((i/total)*100, 2)
      print(f"{precent} percent through processing.")

  with open(os.path.join(generated_output_path, f'predictions_{split}.txt'), 'w') as file:
     for label in predicted_labels:
        file.write(f"{label}\n")

for split in range(0, 5):

  split_id_file = os.path.join(train_test_dir, f'split_{split}')

  with open(split_id_file, 'r') as file:
      train_ids = file.read().split("\n")

  interpretation_train_df = interpretation_df[interpretation_df["section_id"].isin(train_ids)]
  interpretation_test_df = interpretation_df[~interpretation_df["section_id"].isin(train_ids)]

  X_test = interpretation_test_df["paragraph"].to_list()
  y_test = interpretation_test_df["interpretation"].to_list()

  with open(os.path.join(generated_output_path, f'predictions_{split}.txt'), 'r') as file:
    print(file)
    predicted_labels = [line.rstrip() for line in file]

  print(y_test, predicted_labels)
  class_report = classification_report(y_test, predicted_labels, output_dict=True)

  sample_dict = {
      "model": "interpretation_generative",
      "split": split,

      "macro_f1": round(class_report["macro avg"]["f1-score"], 3),
      "macro_precision": round(class_report["macro avg"]["precision"], 3),
      "macro_recall": round(class_report["macro avg"]["recall"], 3),

      "weighted_f1": round(class_report["weighted avg"]["f1-score"], 3),
      "weighted_precision": round(class_report["weighted avg"]["precision"], 3),
      "weighted_recall": round(class_report["weighted avg"]["recall"], 3),

      "1_f1": round(class_report["interpretation"]["f1-score"], 3),
      "1_precision": round(class_report["interpretation"]["precision"], 3),
      "1_recall": round(class_report["interpretation"]["recall"], 3),

      "0_f1": round(class_report["no interpretation"]["f1-score"], 3),
      "0_precision": round(class_report["no interpretation"]["precision"], 3),
      "0_recall": round(class_report["no interpretation"]["recall"], 3),

  }

  new_row = pd.DataFrame(sample_dict, index = [0])
  full_df = pd.concat([full_df, new_row])

  macro_f1_l.append(class_report["macro avg"]["f1-score"])
  macro_precision_l.append(class_report["macro avg"]["precision"])
  macro_recall_l.append(class_report["macro avg"]["recall"])

  weighted_f1_l.append(class_report["weighted avg"]["f1-score"])
  weighted_precision_l.append(class_report["weighted avg"]["precision"])
  weighted_recall_l.append(class_report["weighted avg"]["recall"])

  one_f1_l.append(class_report["interpretation"]["f1-score"])
  one_precision_l.append(class_report["interpretation"]["precision"])
  one_recall_l.append(class_report["interpretation"]["recall"])

  zero_f1_l.append(class_report["no interpretation"]["f1-score"])
  zero_precision_l.append(class_report["no interpretation"]["precision"])
  zero_recall_l.append(class_report["no interpretation"]["recall"])

macro_f1 = sum(macro_f1_l) / len(macro_f1_l)
macro_precision = sum(macro_precision_l) / len(macro_precision_l)
macro_recall = sum(macro_recall_l) / len(macro_recall_l)

weighted_f1 = sum(weighted_f1_l) / len(weighted_f1_l)
weighted_precision = sum(weighted_precision_l) / len(weighted_precision_l)
weighted_recall = sum(weighted_recall_l) / len(weighted_recall_l)

one_f1 = sum(one_f1_l) / len(one_f1_l)
one_precision = sum(one_precision_l) / len(one_precision_l)
one_recall = sum(one_recall_l) / len(one_recall_l)

zero_f1 = sum(zero_f1_l) / len(zero_f1_l)
zero_precision = sum(zero_precision_l) / len(zero_precision_l)
zero_recall = sum(zero_recall_l) / len(zero_recall_l)

model_dict = {
    "model": "descriptive_generative",
    "split": "averages",

    "macro_f1": round(macro_f1, 3),
    "macro_precision": round(macro_precision, 3),
    "macro_recall": round(macro_recall, 3),

    "weighted_f1": round(weighted_f1, 3),
    "weighted_precision": round(weighted_precision, 3),
    "weighted_recall": round(weighted_recall, 3),

    "1_f1": round(one_f1, 3),
    "1_precision": round(one_precision, 3),
    "1_recall": round(one_recall, 3),

    "0_f1": round(zero_f1, 3),
    "0_precision": round(zero_precision, 3),
    "0_recall": round(zero_recall, 3),
}

new_row = pd.DataFrame(model_dict, index = [0])
full_df = pd.concat([full_df, new_row])

full_df.to_csv(os.path.join(output_path, 'gpt_interpretation_results.csv'))





