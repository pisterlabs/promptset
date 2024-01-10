
import numpy as np
import openai
import os
import pandas as pd
from   pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


interp_dir = os.getcwd()
train_test_dir = os.path.join(interp_dir, 'train_test_splits', 'train_test_splits_2')
output_path = os.path.join(interp_dir, 'few_shot_results')
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

prompt_text = "Determine the legal interpretation used in the following passage. Return a single choice from GRAND, FORMAL, or NONE. Here are examples:"
formal_example = "Accepting this point, too, for argument's sake, the question becomes: What did \"discriminate\" mean in 1964? As it turns out, it meant then roughly what it means today: \"To make a difference in treatment or favor (of one as compared with others).\" Webster's New International Dictionary 745 (2d ed. 1954). To \"discriminate against\" a person, then, would seem to mean treating that individual worse than others who are similarly situated. [CITE]. In so-called \"disparate treatment\" cases like today's, this Court has also held that the difference in treatment based on sex must be intentional. See, e.g., [CITE]. So, taken together, an employer who intentionally treats a person worse because of sex—such as by firing the person for actions or attributes it would tolerate in an individual of another sex—discriminates against that person in violation of Title VII."
grand_example = "Respondent's argument is not without force. But it overlooks the significance of the fact that the Kaiser-USWA plan is an affirmative action plan voluntarily adopted by private parties to eliminate traditional patterns of racial segregation. In this context respondent's reliance upon a literal construction of §§ 703 (a) and (d) and upon McDonald is misplaced. See [CITE]. It is a \"familiar rule, that a thing may be within the letter of the statute and yet not within the statute, because not within its spirit, nor within the intention of its makers.\" [CITE]. The prohibition against racial discrimination in §§ 703 (a) and (d) of Title VII must therefore be read against the background of the legislative history of Title VII and the historical context from which the Act arose. See [CITE]. Examination of those sources makes clear that an interpretation of the sections that forbade all race-conscious affirmative action would \"bring about an end completely at variance with the purpose of the statute\" and must be rejected. [CITE]. See [CITE]."
none_example = "The questions are, What is the form of an assignment, and how must it be evidenced? There is no precise form. It may be. by delivery. Briggs v. Dorr, CITE, citing numerous cases; Onion v. Paul, 1 Har. & Johns. 114; Dunn v. Snell, CITE; Titcomb v. Thomas, 5 Greenl. 282. True, it is said it must be on a valuable consideration, with intent to transfer it. But these last are requisites in all assignments, or transfers of securities, negotiable or not. It may be by writing under seal, by writing without seal, by oral declarations, accompanied in all cases by delivery, and on a just consideration. The evidence may be by proof of handwriting and proof of. possession. It may be proved by proving the signature of the payee or obligee on the back, and possession by a third person. 3 Gill & Johns. 218"

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_few_shot_interp(text, formal_example, grand_example, none_example):
    prompt = f"{prompt_text}\n\n###\nText: {formal_example}\nFORMAL\n###\nText: {grand_example}\nGRAND\n###\nText: {none_example}\nNONE\n###\nText: {text}\n"
    response = openai.ChatCompletion.create(
        model='gpt-4',
        max_tokens=4,
        messages = [{'role': 'user', 'content': prompt + text}]
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

for split in range(0, 5):

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
    prediction = get_few_shot_interp(text, formal_example, grand_example, none_example).upper()
    predicted_labels.append(prediction)

    if i % 50 == 0:
      precent = round((i/total)*100, 2)
      print(f"{precent}% through processing.")

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
  y_test = interpretation_test_df["class"].to_list()

  with open(os.path.join(generated_output_path, f'predictions_{split}.txt'), 'r') as file:
    print(file)
    predicted_labels = [line.rstrip() for line in file]

  print(y_test, predicted_labels)
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
    "model": "gpt_examples",
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

full_df.to_csv(os.path.join(output_path, 'gpt_examples.csv'))
