from collections import Counter

import openai
from transformers import pipeline

DEFINITION = """
Classify the dialogue given below into only one of the following 20 classes:
FAM/SOCHX
GENHX
PASTMEDICALHX
CC
PASTSURGICAL
ALLERGY
ROS
MEDICATIONS
ASSESSMENT
EXAM
DIAGNOSIS
DISPOSITION
PLAN
EDCOURSE
IMMUNIZATIONS
IMAGING
GYNHX
PROCEDURES
OTHER_HISTORY
LABS

where GENHX is the history of present illness of the patient
where FAM/SOCHX is the history of illness of their family and their social history.
where PASTMEDICALHX is the past medical history of the patient not including the current illness.
where GYNHX is the history of illness related gynaecology
where ALLERGY is discussion of their allergies/reactions on skin
where LABS are discussions about medical investigation reports from labs
where EDCOURSE is the discussion about administering urgent care

DIALOGUE

{}

\n\n\n
"""

openai.api_key = ""
openai.organization = ""

import time
from transformers import AutoTokenizer
# GPT-3 API
def call_gpt(prompt, model='text-davinci-003', stop=None, temperature=0., top_p=1.0,
        max_tokens=128, majority_at=None):
    num_completions = majority_at if majority_at is not None else 1
    num_completions_batch_size = 5
    
    completions = []
    for i in range(20 * (num_completions // num_completions_batch_size + 1)):
        try:
            requested_completions = min(num_completions_batch_size, num_completions - len(completions))
            if model == 'text-davinci-003':
              ans = openai.Completion.create(
                  model=model,
                  max_tokens=max_tokens,
                  stop=stop,
                  prompt=prompt,
                  temperature=temperature,
                  top_p=top_p,
                  n=requested_completions,
                  best_of=requested_completions)
              completions.extend([choice['text'] for choice in ans['choices']])
            elif model == 'gpt-4' or model == 'gpt3.5-turbo':
              ans = openai.ChatCompletion.create(
                  model=model,
                  max_tokens=max_tokens,
                  stop=stop,
                  messages=[
                      {"role":"system", "content":"""You are a helpful assistant who will help me with the below task"""},
                      {"role":"user", "content": prompt}
                  ],
                  temperature=temperature,
                  top_p=top_p,
                  n=requested_completions)
              completions.extend([choice['message']['content'] for choice in ans['choices']])
            if len(completions) >= num_completions:
                return completions[:num_completions]
        except openai.error.RateLimitError as e:
            print(e)
            print("Calling after {}".format(min(i**2, 60)))
            time.sleep(min(i**2, 60))
    raise RuntimeError('Failed to call GPT API')

def run_gpt(dialogues):
  predictions = []
  for idx, row in enumerate(dialogues):
    prompt = DEFINITION.format(row)
    print("**************************************************************")
    print("Calling GPT for sample {}".format(idx+1))
    response = call_gpt(prompt=prompt, model="gpt-4", stop="\n\n\n", temperature=0.6, top_p=.95, max_tokens=100, majority_at=3)
    counter = Counter(response)
    predictions.append(counter.most_common(1)[0][0])
  return predictions


def run_bert(dialogues):
    tokenizer = AutoTokenizer.from_pretrained("mathury/Bio_ClinicalBERT-finetuned-mediQA", use_fast=True)
    predictions = []
    classifier = pipeline("text-classification", model="mathury/Bio_ClinicalBERT-finetuned-mediQA", tokenizer=tokenizer)
    for idx, row in enumerate(dialogues):
        print("Calling BERT for sample {}".format(idx+1))
        predictions.append(classifier(row, max_length=512, truncation=True, padding=True))
    mapping = {0: 'ALLERGY', 1: 'ASSESSMENT', 2: 'CC', 3: 'DIAGNOSIS', 4: 'DISPOSITION', 5: 'EDCOURSE', 6: 'EXAM', 7: 'FAM/SOCHX', 8: 'GENHX', 9: 'GYNHX', 10: 'IMAGING', 11: 'IMMUNIZATIONS', 12: 'LABS', 13: 'MEDICATIONS', 14: 'OTHER_HISTORY', 15: 'PASTMEDICALHX', 16: 'PASTSURGICAL', 17: 'PLAN', 18: 'PROCEDURES', 19: 'ROS'}
    class_predictions = []
    for pred in predictions:
        class_predictions.append(mapping[int(pred[0]['label'].split("_")[1])])
    return class_predictions


def clean_gpt_predictions(predictions):
    cleaned_predictions = []
    for pred in predictions:
        if pred == "SOCHX":
            cleaned_predictions.append("FAM/SOCHX")
        elif "FAM/SOCHX" in pred:
            cleaned_predictions.append("FAM/SOCHX")
        elif "GENHX" in pred:
            cleaned_predictions.append("GENHX")
        elif "GYNHX" in pred:
            cleaned_predictions.append("GYNHX")
        elif "MEDICATIONS" in pred:
            cleaned_predictions.append("MEDICATIONS")
        elif "PLAN" in pred:
            cleaned_predictions.append("PLAN")
        elif "PASTMEDICALHX" in pred:
            cleaned_predictions.append("PASTMEDICALHX")
        elif "DIAGONOSIS" in pred:
            cleaned_predictions.append("DIAGNOSIS")
        elif "ROS" in pred:
            cleaned_predictions.append("ROS")
        elif "ASSESSMENT" in pred:
            cleaned_predictions.append("ASSESSMENT")
        elif "DISPOSITION" in pred:
            cleaned_predictions.append("DISPOSITION")
        elif "EDCOURSE" in pred:
            cleaned_predictions.append("EDCOURSE")
        elif "EXAM" in pred:
            cleaned_predictions.append("EXAM")
        elif "IMAGING" in pred:
            cleaned_predictions.append("IMAGING")
        elif "IMMUNIZATIONS" in pred:
            cleaned_predictions.append("IMMUNIZATIONS")
        elif "LABS" in pred:
            cleaned_predictions.append("LABS")
        elif "OTHER_HISTORY" in pred:
            cleaned_predictions.append("OTHER_HISTORY")
        elif "PAST_SURGICAL" in pred:
            cleaned_predictions.append("PAST_SURGICAL")
        elif "PLAN" in pred:
            cleaned_predictions.append("PLAN")
        elif "PROCEDURES" in pred:
            cleaned_predictions.append("PROCEDURES")
        elif "ALLERGY" in pred:
            cleaned_predictions.append("ALLERGY")
        else:
            cleaned_predictions.append(pred)
    return cleaned_predictions




def run_task_A_classification(dialogues):
    gpt_predictions = run_gpt(dialogues)
    print(gpt_predictions)
    cleaned_gpt_predictions = clean_gpt_predictions(gpt_predictions)
    print(cleaned_gpt_predictions)
    bert_predictions = run_bert(dialogues)
    print(bert_predictions)
    chosen_predictions = []
    for bb, gpt in zip(bert_predictions, cleaned_gpt_predictions):
        if bb == "ROS" or bb == "GENHX" or bb == "CC":
            chosen_predictions.append(bb)
        else:
            chosen_predictions.append(gpt)
    return chosen_predictions
