import openai
import random
from collections import Counter
import string
import re
import argparse
import json
import datasets
import pickle
import tiktoken
from typing import List
import numpy as np


def get_model_context_limits(name):
    limits = {
        'gpt-4': 8192,
        'gpt-4-0613': 8192,
        'gpt-4-32k-0613': 32768,
        'gpt-4-32k': 32768,
        'gpt-3.5-turbo': 4096,
        'gpt-3.5-turbo-16k': 16384,
        'gpt-3.5-turbo-0613': 4096,
        'gpt-3.5-turbo-16k-0613': 16384,
        'text-davinci-003': 4097,
        'text-davinci-002': 4097,
        'code-davinci-002': 8001,
        'text-curie-001': 2049,
        'text-babbage-001': 2049,
        'text-ada-001': 2049,
        'davinci': 2049,
        'curie': 2049,
        'babbage': 2049,
        'ada': 2049
    }
    return limits[name]


def get_inputs(setting, api_in_prompt):
    assert setting in ["zeroshot", "fewshot", "maxshot", "maxshotinstruction"]

    if setting == "zeroshot":
        num_exs = 0
        instruction_idx = 0
        instruction = 'Generate a function with excel script to execute the action given below in natural language'
    if setting == "fewshot":
        num_exs = 3
        instruction_idx = 0
        instruction = 'Generate a function with excel script to execute the action given below in natural language'
    if setting == "maxshot":
        num_exs = 10
        instruction_idx = 0
        instruction = 'Generate a function with excel script to execute the action given below in natural language'
    if setting == "maxshotinstruction":
        num_exs = 10
        instruction_idx = 1
        instruction = 'Generate a function with excel script to execute the action given below in NL. You also need to generate comment describing the operation you are performing. Make sure to generate a valid excel operation and pass appropriate parameters as provided in the action information. Simple solution is preferred over a complex one'

    if api_in_prompt:
        with open("abbrev_API.csv", 'r') as f:
            instruction += "\nUse the following API. M stands for Method, E stands for EnumField, C stands for Class, P stands for Property.\n===BEGIN API===\n" + f.read() + "\n===END API==="

    return num_exs, instruction_idx, instruction


def rouge(prediction: str, ground_truth: str):
    score = rouge_metric.compute(
        predictions=[prediction],
        references=[ground_truth],
        **{'use_stemmer': True, 'rouge_types': ['rougeL']}
    )
    return score['rougeL'][0].fmeasure


def dynamic_sample(num_exs, query, traindatali):
    if num_exs:
        similarities = []
        for ex in traindatali:
            try:
                similarities.append(f1_score(query, ex['input']))
            except:
                similarities.append(0)
        return np.argsort(similarities)[::-1][:num_exs]


def sacrebleu(prediction: str, ground_truth_list: List[str]):
    score = sacrebleu_metric.compute(
        predictions=[prediction],
        references=[ground_truth_list]  # scarebleu expects several golds per
    )
    return score['score'] / 100


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction: str, ground_truth: str):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def remove_params(input_str):
    output_str = ""
    buffer = ""
    st_idx = -1
    for idx, c in enumerate(input_str):
        if c not in ["(", ")"]:
            buffer += c
        elif c == "(":
            output_str += buffer + "("
            st_idx = idx
            buffer = ""
        elif c == ")" and st_idx > -1:
            output_str += "<PARAMS>)"
            st_idx = -1
            buffer = ""
        elif c == ")" and st_idx == -1:
            output_str += buffer + ")"
            buffer = ""

    output_str = re.sub(r'[A-Z]\d+', '<RANGE>', output_str)
    output_str = re.sub(r'\d+(\.\d*)?', '<NUMBER>', output_str)
    return output_str


def cut_down_data(data_list, non_data_prompt_material, tokenizer_name, context_percentage):
    # Construct what the prompt would look like, then see if it exceeds the token limit. If not, return.
    # If yes, strip out half of each sheet in the data.
    current_prompt = " ".join(data_list) + " " + non_data_prompt_material
    enc = tiktoken.encoding_for_model(tokenizer_name)
    num_tokens = len(enc.encode(current_prompt))
    context_size = get_model_context_limits(tokenizer_name)
    while num_tokens > context_percentage*context_size:
        print(num_tokens)
        # Cut out the bottom half of each sheet in the data
        new_data_list = []
        for d in data_list:
            new_excel_workbook = []
            sheets = d.split("SHEETNAME: ")[1:]
            for s in sheets:
                if len(s) > 1:
                    s = s[:len(s)//2]
                new_excel_workbook.append("SHEETNAME: " + s)
            new_data_list.append("".join(new_excel_workbook))
        current_prompt = " ".join(new_data_list) + " " + non_data_prompt_material
        new_num_tokens = len(enc.encode(current_prompt))
        if new_num_tokens == num_tokens:
            return None
        num_tokens = len(enc.encode(current_prompt))
        data_list = new_data_list
    return data_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument("--setting", type=str)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--openai_key", type=str)
    parser.add_argument('--api_in_prompt', dest='api_in_prompt', default=False, action='store_true')
    parser.add_argument('--dynamic_prompt', dest='dynamic_prompt', default=False, action='store_true')
    return parser.parse_args()


# Settings should be "zeroshot", "fewshot", "maxshot", and "maxshotinstructions".
# Models should be gpt-3.5-turbo-16k and gpt-4-32k
# Tokenizer_name should be gpt-3.5-turbo-16k and gpt-4-32k
if __name__ == "__main__":
    args = parse_args()
    model_name = args.model
    tokenizer_name = args.tokenizer_name
    setting = args.setting
    seed = args.seed
    openai.api_key = args.openai_key
    api_in_prompt = args.api_in_prompt
    dynamic_prompt = args.dynamic_prompt

    rouge_metric = datasets.load_metric('rouge')
    sacrebleu_metric = datasets.load_metric('sacrebleu')

    num_exs, instruction_idx, instruction = get_inputs(setting, api_in_prompt)

    random.seed(num_exs + 10 * instruction_idx)

    pred_filename = model_name + "_" + tokenizer_name + "_" + setting + "_" + str(api_in_prompt) + "_" + str(dynamic_prompt) + '.json'
    eval_filename = model_name + "_" + tokenizer_name + "_" + setting + "_" + str(api_in_prompt) + "_" + str(dynamic_prompt) + '.eval.pkl'

    # Load in the training set examples
    with open('full_bench_train.json', 'r') as f:
        traindatali = json.load(f)

    promptex = ''
    indexlist = []
    while len(indexlist) < num_exs:
        ranin = random.randint(200, len(traindatali) - 1)
        if ranin not in indexlist and ranin > 199:
            indexlist.append(ranin)

    code_header = "function main(workbook: ExcelScript.Workbook) {  let selectedSheet = workbook.getActiveWorksheet(); // "

    big_prompt = False
    if num_exs:
        non_data_prompt_material = instruction + " " + " ".join([traindatali[i]["input"] for i in indexlist]) + \
                                   " ".join([traindatali[i]['output'] for i in indexlist]) + \
                                   "\nExcel Script Function:\n" + '\nData:\n' + '\nAction:\n'
        stripped_data = cut_down_data([traindatali[i]['data_string'] for i in indexlist],
                                      non_data_prompt_material,
                                      tokenizer_name,
                                      .8)
        if stripped_data is None:
            big_prompt = True
            stripped_data = [traindatali[idx]['data_string'] for idx in indexlist]

        promptex = ''
        for inum, i in enumerate(indexlist):
            promptex += '\nAction:\n' + str(traindatali[i]['input']) + '\nData:\n' + str(stripped_data[inum]) + \
                        '\nExcel Script Function:\n' + str(traindatali[i]['output'])

        if big_prompt:
            enc = tiktoken.encoding_for_model(tokenizer_name)
            promptex = enc.decode(enc.encode(promptex)[:int(get_model_context_limits(tokenizer_name)*.8)])

    with open('full_bench_test.json', 'r') as f:
        datali = json.load(f)

    responselist = []
    promptlist = []
    goldlist = []
    excelfilelist = []
    exceldesclist = []

    num_eval = 200

    for i in range(num_eval):
        if i % 10 == 0:
            print("Processing test example %d of %d" % (i, num_eval))

        if dynamic_prompt:
            indexlist = dynamic_sample(num_exs, str(datali[i]['input']), traindatali)
            # Construct the prompt using these indices, same way as above
            big_prompt = False
            if num_exs:
                non_data_prompt_material = instruction + " " + " ".join([traindatali[idx]["input"] for idx in indexlist]) + \
                                           " ".join([traindatali[idx]['output'] for idx in indexlist]) + \
                                           "\nExcel Script Function:\n" + '\nData:\n' + '\nAction:\n'
                stripped_data = cut_down_data([traindatali[idx]['data_string'] for idx in indexlist],
                                              non_data_prompt_material,
                                              tokenizer_name,
                                              .8)
                if stripped_data is None:
                    big_prompt = True
                    stripped_data = [traindatali[idx]['data_string'] for idx in indexlist]

                promptex = ''
                for inum, idx in enumerate(indexlist):
                    promptex += '\nAction:\n' + str(traindatali[idx]['input']) + '\nData:\n' + str(stripped_data[inum]) + \
                                '\nExcel Script Function:\n' + str(traindatali[idx]['output'])

                if big_prompt:
                    enc = tiktoken.encoding_for_model(tokenizer_name)
                    promptex = enc.decode(enc.encode(promptex)[:int(get_model_context_limits(tokenizer_name) * .8)])

        prompt_final = instruction
        prompt_final += promptex
        prompt_final += '\n' + 'Action:' + '\n' + str(datali[i]['input'])
        stripped_data = cut_down_data([str(datali[i]['data_string'])], prompt_final + '\nData:\n' + '\nExcel Script Function:\n' + code_header, tokenizer_name, 1)
        if stripped_data is None:
            stripped_data = [""]
        prompt_final += '\nData:\n' + str(stripped_data[0])
        prompt_final += '\nExcel Script Function:\n' + code_header
        promptlist.append(prompt_final)

        excelfilelist.append(datali[i]['metadata']['public_url'])
        exceldesclist.append(datali[i]['metadata']['filedescription'])
        goldlist.append(str(datali[i]['output']))

        if model_name != "dummy":
            response = openai.Completion.create(model=model_name, prompt=prompt_final, temperature=0,
                                                top_p=1, frequency_penalty=0, presence_penalty=0, max_tokens=256)
        else:
            response = {"choices": [{"text": "Set range D254 on selectedSheet selectedSheet.getRange(\"D254\").setFormulaLocal(\"=xor(C253)\"); }"}]}
        responselist.append(response)
    output_list = []
    for i in range(len(responselist)):
        output_list.append({
            "output": code_header +
                      responselist[i]['choices'][0]['text'],
            "input": promptlist[i],
            "goldoutput": goldlist[i], "excelfilelink": excelfilelist[i],
            "excelfiledesc": exceldesclist[i]})
    prediction_dict = {"predictions": output_list}
    structl = ''
    with open(pred_filename, 'w') as outfile:
        json.dump(prediction_dict, outfile)



    with open(pred_filename) as f:
        d = json.load(f)

    print("Predictions written to", pred_filename)
    print("Running evaluations")
    metrics = {}
    max_eval = 200
    for i in range(max_eval):
        gold_outputs = []
        gold_outputs_minus_params = []
        gold_outputs.append(d['predictions'][i]['goldoutput'][103:])
        gold_outputs_minus_params.append(remove_params(gold_outputs[-1]))
        pred = d['predictions'][i]['output'][103:]
        pred_minus_params = remove_params(pred)

        # long-range text generation metrics
        if 'rouge' not in metrics:
            metrics['sacrebleu'] = metrics['rouge'] = metrics['exact_match'] = metrics['f1'] = metrics['sacrebleu_minus_params'] = metrics['rouge_minus_params'] = metrics['exact_match_minus_params'] = metrics['f1_minus_params'] = 0
        metrics['rouge'] += metric_max_over_ground_truths(rouge, pred, gold_outputs)
        metrics['sacrebleu'] += sacrebleu(pred, gold_outputs)
        metrics['exact_match'] += metric_max_over_ground_truths(exact_match_score, pred, gold_outputs)
        metrics['f1'] += metric_max_over_ground_truths(f1_score, pred, gold_outputs)
        metrics['rouge_minus_params'] += metric_max_over_ground_truths(rouge, pred_minus_params, gold_outputs_minus_params)
        metrics['sacrebleu_minus_params'] += sacrebleu(pred_minus_params, gold_outputs_minus_params)
        metrics['exact_match_minus_params'] += metric_max_over_ground_truths(exact_match_score, pred_minus_params, gold_outputs_minus_params)
        metrics['f1_minus_params'] += metric_max_over_ground_truths(f1_score, pred_minus_params, gold_outputs_minus_params)

    # normalize tne metrics
    for key in metrics.keys():
        metrics[key] /= min(200, max_eval)
        metrics[key] *= 100

    print("Evaluation complete. Evaluation metrics printed below, and saved in", eval_filename)
    print("Setting: " + setting)
    print("Model: " + model_name)
    context_size = get_model_context_limits(tokenizer_name)
    print("Context size: " + str(context_size))
    print(metrics)

    with open(eval_filename, 'wb') as f:
        pickle.dump(metrics, f)

