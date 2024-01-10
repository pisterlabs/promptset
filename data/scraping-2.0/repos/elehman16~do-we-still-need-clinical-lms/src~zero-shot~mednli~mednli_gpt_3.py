import json
import argparse
import pandas as pd
import numpy as np 
from sklearn.metrics import f1_score, accuracy_score
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

import openai
from sklearn.utils import shuffle

openai.api_key = ""
openai.api_base =  ""
openai.api_type = ""
openai.api_version = ""
deployment_id=""

def get_mapping() -> dict:
    mapper1 = {'yes': 'entailment', 'no': 'contradiction', 'maybe': 'neutral', 'it is not possible to tell': 'neutral'}
    mapper2 = {'true': 'entailment', 'false': 'contradiction', 'inconclusive': 'neutral'}
    mapper3 = {'always': 'entailment', 'never': 'contradiction', 'sometimes': 'neutral'}
    mapper4 = {'entailment': 'entailment', 'contradiction': 'contradiction', 'neutral': 'neutral'}
    mapper5 = {'true': 'entailment', 'false': 'contradiction', 'neither': 'neutral'}  
    mapper6 = {'yes': 'entailment', 'no': 'contradiction', "it's impossible to say": 'neutral', 'it is not possible to say': 'neutral'} 
    
    combiner = {}
    mappers = [mapper1, mapper2, mapper3, mapper4, mapper5, mapper6]
    for m in mappers:
        combiner = combiner | m

    return combiner

def format_single_example(sentence1: str, sentence2: str, prompt: str) -> str:
    if prompt == '0':
        return f"Answer entailment, neutral or contradiction.\n\nPremise: {sentence1}\nHypothesis: {sentence2}\nAnswer:"
    elif prompt == '1':
        return f'Suppose {sentence1} Can we infer that "{sentence2}"? Yes, no, or maybe?'
    elif prompt == '2':
        return f'{sentence1} Based on that information, is the claim: "{sentence2}" true, false, or inconclusive?'
    elif prompt == '3':
        return f'Given that "{sentence1}". Does it follow that "{sentence2}" Yes, no, or maybe?'
    elif prompt == '4':
        return f'Suppose it\'s true that {sentence1} Then, is "{sentence2}" always, sometimes, or never true?'
    elif prompt == '5':
        prompt = 'Answer entailment, contradiction or neutral.\nPremise: Labs were notable for Cr 1.7 (baseline 0.5 per old records) and lactate 2.4.\nHypothesis: Patient has elevated Cr\nAnswer: entailment\n\nPremise: Labs were notable for Cr 1.7 (baseline 0.5 per old records) and lactate 2.4.\nHypothesis: Patient has elevated BUN\nAnswer: neutral\n\nPremise: Labs were notable for Cr 1.7 (baseline 0.5 per old records) and lactate 2.4.\nHypothesis: Patient has normal Cr\nAnswer: contradiction\n\n'
        prompt += f'Premise: {sentence1}\nHypothesis: {sentence2}\nAnswer:'
        return prompt
    elif prompt == '6':
        prompt = 'Premise: Labs were notable for Cr 1.7 (baseline 0.5 per old records) and lactate 2.4.\nHypothesis: Patient has elevated Cr\nAnswer entailment, contradiction or neutral: entailment\n\nPremise: Labs were notable for Cr 1.7 (baseline 0.5 per old records) and lactate 2.4.\nHypothesis: Patient has elevated BUN\nAnswer entailment, contradiction or neutral: neutral\n\nPremise: Labs were notable for Cr 1.7 (baseline 0.5 per old records) and lactate 2.4.\nHypothesis: Patient has normal Cr\nAnswer entailment, contradiction or neutral: contradiction\n\n'
        prompt += f'Premise: {sentence1}\nHypothesis: {sentence2}\nAnswer entailment, contradiction or neutral:'
        return prompt
    elif prompt == '7':
        prompt = f"{sentence1} Question: {sentence2} True, False, or Neither?"
        return prompt
    elif prompt == '8':
        prompt = 'Given that "Labs were notable for Cr 1.7 (baseline 0.5 per old records) and lactate 2.4.". Does it follow that "Patient has elevated Cr" Yes, no, or maybe? Yes.\n\nGiven that "Labs were notable for Cr 1.7 (baseline 0.5 per old records) and lactate 2.4.". Does it follow that "Patient has elevated BUN" Yes, no, or maybe? Maybe.\n\nGiven that "Labs were notable for Cr 1.7 (baseline 0.5 per old records) and lactate 2.4.". Does it follow that "Patient has normal Cr" Yes, no, or maybe? No.\n\n' 
        prompt += f'Given that "{sentence1}". Does it follow that "{sentence2}" Yes, no, or maybe?'
        return prompt
    elif prompt == '9':
        prompt = 'Given that "It was not associated with any shortness of breath, nausea, vomiting, or she tried standing during this episode.” Does it follow that "She had vomiting and dyspnea with this episode”? Yes, no, or maybe? No.\n\nGiven that "He has been followed by Dr. [**Last Name (STitle) 21267**] of Podiatry for chronic first toe MTP ulcer, which is now resolving.” Does it follow that "He had a chronic wound on his toe”? Yes, no, or maybe? Yes.\n\nGiven that "She had no fevers/chills/sweats prior to coming to the hosptial.” Does it follow that "Patient has a normal abdominal CT”? Yes, no, or maybe? Neutral.'
        prompt += f'Given that {sentence1} Does it follow that {sentence2.strip()}? Yes, no, or maybe?'
        return prompt 
    elif prompt == '10':
        prompt = f"Does {sentence1} mean that {sentence2}?\n\n Options:\n-Yes\n-No\n-It's impossible to say"
        return prompt
    elif prompt == '11':
        if sentence1.strip()[-1] == '.':
            sentence1 = sentence1[:-1]

        prompt = f'Does "{sentence1}" mean that "{sentence2.strip()}"?\n\nOptions:\n-Yes\n-No\n-It\'s impossible to say'
        return prompt

    elif prompt == '12':
        if sentence1.strip()[-1] == '.':
            sentence1 = sentence1[:-1]

        prompt = 'Does “Labs were notable for Cr 1.7 (baseline 0.5 per old records) and lactate 2.4” mean that "Patient has elevated Cr"?\n\nOptions:\n-Yes\n-No\n-It\'s impossible to say \nYes\n\nDoes “Labs were notable for Cr 1.7 (baseline 0.5 per old records) and lactate 2.4” mean that "Patient has elevated BUN"?\n\nOptions:\n-Yes\n-No\n-It\'s impossible to say \nNo\n\nDoes “Labs were notable for Cr 1.7 (baseline 0.5 per old records) and lactate 2.4” mean that "Patient has normal Cr"?\n\nOptions:\n-Yes\n-No\n-It\'s impossible to say \nIt’s impossible to say\n\n'
        actual_prompt = f'Does "{sentence1}" mean that "{sentence2.strip()}"?\n\nOptions:\n-Yes\n-No\n-It\'s impossible to say'
        return prompt + actual_prompt
    elif prompt == '13':
        if sentence1.strip()[-1] == '.':
            sentence1 = sentence1[:-1]

        prompt = 'Does “It was not associated with any shortness of breath, nausea, vomiting, or she tried standing during this episode” mean that "She had vomiting and dyspnea with this episode"?\n\n Options:\n-Yes\n-No\n-It\'s impossible to say \nNo\n\nDoes “He has been followed by Dr. [**Last Name (STitle) 21267**] of Podiatry for chronic first toe MTP ulcer, which is now resolving” mean that "He had a chronic wound on his toe"?\n\n Options:\n-Yes\n-No\n-It\'s impossible to say \nYes\n\nDoes “She had no fevers/chills/sweats prior to coming to the hosptial” mean that "Patient has a normal abdominal CT"?\n\n Options:\n-Yes\n-No\n-It\'s impossible to say \nIt’s impossible to say'
        actual_prompt = f'Does "{sentence1}" mean that "{sentence2.strip()}"?\n\nOptions:\n-Yes\n-No\n-It\'s impossible to say'
        return prompt + actual_prompt
    
    elif prompt == '14':
        prompt = f"Does Discharge Summary: {sentence1} mean that {sentence2}?\n\n Options:\n-Yes\n-No\n-It's impossible to say"
        return prompt
    else:
        return ""

def preprocess_function(examples, prompt: str) -> tuple[list, list]:
    """Format the examples and then tokenize them. """
    inputs = [format_single_example(s1, s2, prompt) for s1, s2 in zip(examples['sentence1'], examples['sentence2'])]
    targets = examples['gold_label']
    return inputs, targets

def read_jsonl(file_path: str):
    """Read the given JSONL file."""
    with open(file_path) as f:
        data = [json.loads(line) for line in f]
    
    return data

def get_data(mednli_path: str):
    """Get the mednli data. """
    # mli_dev_v1.jsonl  mli_test_v1.jsonl  mli_train_v1.jsonl
    train = Dataset.from_list(read_jsonl(mednli_path + '/mli_train_v1.jsonl'))
    val = Dataset.from_list(read_jsonl(mednli_path + '/mli_dev_v1.jsonl'))
    test = Dataset.from_list(read_jsonl(mednli_path + '/mli_test_v1.jsonl'))
    return DatasetDict({"train": train, "val": val, "test": test})

def query_openai_all(all_inputs):
    """Query OpenAI. """
    all_answers = []
    for i in range(0, len(all_inputs), 20):
        response = openai.Completion.create(engine=deployment_id, prompt=all_inputs[i:i+20], max_tokens=256, temperature=0)
        all_answers.extend([resp['text'].strip() for resp in response['choices']])

    return all_answers

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mednli-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--prompt', type=str)
    args = parser.parse_args()

    # Get data and use the tokenizer on the data 
    dataset_dict = get_data(args.mednli_dir)
    test_inputs, test_targets = preprocess_function(dataset_dict['test'], args.prompt)

    # Test the model.
    predictions = query_openai_all(dataset_dict['test'])
    with open(args.output_dir + '/predictions.json', 'w') as f:
        json.dump(predictions, f)