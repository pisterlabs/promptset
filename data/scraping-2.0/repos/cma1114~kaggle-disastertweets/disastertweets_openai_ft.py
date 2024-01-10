import os
import pandas as pd
#from sklearn.model_selection import train_test_split
from datasets import Dataset
import openai
import re
import json
from config import OPENAI_API_KEY, HOME_DIR

"""
#format the data for openai
df = pd.read_csv(HOME_DIR+"/input/train_cln.csv")
df['keyword'] = df['keyword'].fillna('unknown')
df['location'] = df['location'].fillna('unknown')
sep = "[SEP]"
df['text'] = df.apply(lambda row: f"{row['text']} {sep} keyword: {row['keyword']} {sep} location: {row['location']}", axis=1)
df['text'] = df['text'].str.replace('->', '-') 

df['prompt'] = df['text'] + ' ->'
df['completion'] = df['target'].apply(lambda x: ' Real disaster' if x == 1 else ' Not a real disaster')

ds = Dataset.from_pandas(df)
ds = ds.train_test_split(test_size=0.05, seed=42)
train_df = ds['train'].to_pandas()
val_df = ds['test'].to_pandas()
train_df[['prompt', 'completion']].to_csv(HOME_DIR+"/input/train_openai_cln.csv", index=False)
val_df[['prompt', 'completion']].to_csv(HOME_DIR+"/input/val_openai_cln.csv", index=False)

test_df = pd.read_csv(HOME_DIR+"/input/test.csv")
test_df['keyword'] = test_df['keyword'].fillna('unknown')
test_df['location'] = test_df['location'].fillna('unknown')
sep = "[SEP]"
test_df['text'] = test_df.apply(lambda row: f"{row['text']} {sep} keyword: {row['keyword']} {sep} location: {row['location']}", axis=1)
test_df['text'] = test_df['text'].str.replace('->', '-') 
test_df['prompt'] = test_df['text'] + ' ->'
test_df[['prompt']].to_csv("/Users/christopherackerman/input/test_openai_cln.csv", index=False)
"""

"""
#call OpenAI to fine tune the model
#on command line:
#https://platform.openai.com/docs/guides/fine-tuning
openai tools fine_tunes.prepare_data -f {HOME_DIR}+/input/train_openai_cln.csv
#After you’ve fine-tuned a model, remember that your prompt has to end with the indicator string ` ->` for the model to start generating completions, rather than continuing with the prompt. Make sure to include `stop=["eal disaster"]` so that the generated texts ends at the expected place.
openai api fine_tunes.create -t "./input/train_openai_cln_prepared.jsonl" -m curie
###openai api fine_tunes.create -t "train_openai_prepared_train.jsonl" -v "train_openai_prepared_valid.jsonl" --compute_classification_metrics --classification_positive_class " 0" -m "curie"
####After you’ve fine-tuned a model, remember that your prompt has to end with the indicator string `\n\n###\n\n` for the model to start generating completions, rather than continuing with the prompt.
#openai api fine_tunes.list
#openai api fine_tunes.follow -i ft-xrr8yar4nGUWGWhTQPznSXMB
"""

#now use the fine tuned model for classification
openai.api_key = OPENAI_API_KEY
model_name="curie:ft-personal-2023-06-20-03-24-52"
#suffix = " ->"
stopmarker = "eal disaster"

###do eval set
df = pd.read_csv(HOME_DIR+"/input/val_openai_cln.csv")

# Initialize the count of correct predictions
correct_predictions = 0

# Initialize a list to store the text responses
responses = []

# Iterate through the validation set
#for item in data:
for index, item in df.iterrows():
    response = openai.Completion.create(
        model=model_name,
        prompt=item["prompt"],# + suffix,
        max_tokens=20,#1,
        n=1,
        stop=stopmarker,#None,#["eal disaster"],#
        temperature=0.0,  # Lower temperature for more deterministic output
    )
    predicted_label = response.choices[0].text.strip().lower() + stopmarker
    true_label = item["completion"].strip().lower()
    print("predicted_label", predicted_label)
    print("true_label", true_label)

    # Compare the predicted label with the true label
    if predicted_label == true_label:
        correct_predictions += 1
#        print("correct_predictions", correct_predictions)

    # Store the text response along with the input text and predicted label
    responses.append({
        "input_text": item["prompt"],
        "predicted_label": predicted_label,
        "true_label": true_label
    })


# Calculate and print the accuracy
accuracy = correct_predictions / len(df)
print(f"Correct predictions: {correct_predictions}")
print(f"Total predictions: {len(df)}")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the responses to a file
responses_df = pd.DataFrame(responses)
responses_df['input_text'] = responses_df['input_text'].apply(lambda x: x[:-3] if x.endswith(' ->') else x)
responses_df['predicted_label'] = responses_df['predicted_label'].apply(lambda x: 0 if x == 'not a real disaster' else 1)
responses_df['true_label'] = responses_df['true_label'].apply(lambda x: 0 if x == 'not a real disaster' else 1)

#get id from orig file - first need to do the same preproc on text so you can join on it
first_df = pd.read_csv(HOME_DIR+"/input/train_cln.csv")
first_df['keyword'] = first_df['keyword'].fillna('unknown')
first_df['location'] = first_df['location'].fillna('unknown')
sep = " [SEP] "
first_df['text'] = first_df.apply(lambda row: f"{row['text']} {sep} keyword: {row['keyword']} {sep} location: {row['location']}", axis=1)
first_df['text'] = first_df['text'].str.replace('->', '-') 
merged_df = pd.merge(responses_df, first_df[['text', 'id']], left_on='input_text', right_on='text', how='left')
merged_df = merged_df.drop(columns='text')
merged_df.to_csv("gptcurie_finetuned_aug_cln_eval_short.csv", index=False)
#responses_df.to_csv("responses_gptfinetuned_aug_cln_valid.csv", index=False)


### now do test set
df = pd.read_csv(HOME_DIR+"/input/test_openai_cln.csv")

# Initialize a list to store the text responses
responses = []

# Iterate through the validation set
#for item in data:
for index, item in df.iterrows():
    response = openai.Completion.create(
        model=model_name,
        prompt=item["prompt"],# + suffix,
        max_tokens=20,#1,
        n=1,
        stop=stopmarker,#None,#["eal disaster"],#
        temperature=0.0,  # Lower temperature for more deterministic output
    )
    predicted_label = response.choices[0].text.strip().lower() + stopmarker
    print("predicted_label", predicted_label)

    # Store the text response along with the input text and predicted label
    responses.append({
        "input_text": item["prompt"],
        "predicted_label": predicted_label
    })


# Save the responses to a file
responses_df = pd.DataFrame(responses)
responses_df['input_text'] = responses_df['input_text'].apply(lambda x: x[:-3] if x.endswith(' ->') else x)
responses_df['predicted_label'] = responses_df['predicted_label'].apply(lambda x: 0 if x == 'not a real disaster' else 1)

first_df = pd.read_csv(HOME_DIR+"/input/test.csv")
first_df['keyword'] = first_df['keyword'].fillna('unknown')
first_df['location'] = first_df['location'].fillna('unknown')
sep = " [SEP] "
first_df['text'] = first_df.apply(lambda row: f"{row['text']} {sep} keyword: {row['keyword']} {sep} location: {row['location']}", axis=1)
first_df['text'] = first_df['text'].str.replace('->', '-') 
merged_df = pd.merge(responses_df, first_df[['text', 'id']], left_on='input_text', right_on='text', how='left')
merged_df = merged_df.drop(columns='text')
merged_df.to_csv("gptcurie_finetuned_aug_cln_test.csv", index=False)
