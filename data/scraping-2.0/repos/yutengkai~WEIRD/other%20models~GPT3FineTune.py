pip install --upgrade openai

import openai
import os
import pandas as pd
from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report
import matplotlib.pyplot as plt
import math

os.environ['OPENAI_API_KEY'] = "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"
openai.api_key = "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"

drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/DeClare/bert_set.csv').drop('Unnamed: 0', axis=1)

def prompt_completion_format(df):
  # df['prompt'] = 'Is the following claim representing the information correctly from the following text?' + '\nClaim: ' + df['claim'] + '\nText: ' + df['text'] + '\n\n###\n\n'
  df['prompt'] = 'Text: ' + df['text'] + '\nClaim: ' + df['claim'] + '\nSupported:'
  df = df.drop('claim', axis=1)
  df = df.drop('text', axis=1)
  df.rename(columns={'prompt': 'prompt', 'label': 'completion'}, inplace=True)
  df['completion'] = df['completion'].map({True: " yes", False: " no"})
  df = df[['prompt', 'completion']]
  return df

df = prompt_completion_format(df)

train_df, val_test_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=42)

train_df.to_json("gpt_train_set.jsonl", orient='records', lines=True)
val_df.to_json("gpt_valid_set.jsonl", orient='records', lines=True)

# !openai tools fine_tunes.prepare_data -f gpt_train_set.jsonl -q
!openai api fine_tunes.create -t "gpt_train_set.jsonl" -v "gpt_valid_set.jsonl" --compute_classification_metrics --classification_positive_class " yes" -m ada
!openai api fine_tunes.follow -i ft-JQ3F1OEncjm2l8qaeruyd69D

!openai api fine_tunes.results -i ft-JQ3F1OEncjm2l8qaeruyd69D > result.csv
results = pd.read_csv('result.csv')
results[results['classification/accuracy'].notnull()].tail(1)

main_test_df = pd.read_csv('/content/drive/MyDrive/DeClare/test_set.csv').drop('Unnamed: 0', axis=1)
main_test_df = prompt_completion_format(main_test_df)
# main_test_df.to_json("gpt_test_set.jsonl", orient='records', lines=True)
# main_test_df = pd.read_json('gpt_test_set.jsonl', lines=True)

main_test_df.head()
print(main_test_df['prompt'][3])
print(main_test_df['completion'][3])

# ft_model = 'ada:ft-fake-news-detection-project-2023-05-27-08-29-18'
# ft_model = 'ada:ft-fake-news-detection-project-2023-05-28-05-18-21'
# ft_model = 'ada:ft-fake-news-detection-project-2023-05-30-05-15-00'
ft_model = 'ada:ft-fake-news-detection-project-2023-05-30-07-42-22'
res = openai.Completion.create(model=ft_model, prompt=main_test_df['prompt'][3], max_tokens=1, logprobs=2)
# res.choices[0]['text']
res

ground_truth_labels = main_test_df['completion'].tolist()
predicted_labels = []
predicted_probs = []
for i in range(len(main_test_df)):
  res = openai.Completion.create(model=ft_model, prompt=main_test_df['prompt'][i], max_tokens=1, logprobs=2)
  predicted_labels.append(res.choices[0]['text'])
  predicted_probs.append(math.exp(float(res.choices[0]['logprobs']['token_logprobs'][0])))

ground_truth_labels = [1 if label==' yes' else 0 for label in ground_truth_labels]
predicted_labels = [1 if label==' yes' else 0 for label in predicted_labels]
for i in range(len(predicted_probs)):
  if predicted_labels[i] == 0:
    predicted_probs[i] = 1.0 - predicted_probs[i]

# Classification report
print(classification_report(ground_truth_labels, predicted_labels))

# Calculate ROC AUC
fpr, tpr, _ = roc_curve(ground_truth_labels, predicted_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Calculate PR AUC
precision, recall, _ = precision_recall_curve(ground_truth_labels, predicted_probs)
pr_auc = auc(recall, precision)

# Plot PR curve
plt.figure()
plt.plot(recall, precision, color='darkorange', label='PR curve (area = %0.2f)' % pr_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="lower right")
plt.show()
