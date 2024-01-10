import json
import os
import pickle
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pprint import pprint

import openai
from sklearn.metrics import precision_recall_fscore_support
from tongueswitcher_evaluation import *
from tqdm import tqdm

test_cases = {}

CLH_TEST_CASES_FILE = "../tongueswitcher-corpus/interlingual_homograph_testset.jsonl"
FLAIR_CACHE_FILE = "../data/cache/clh_flair_cache.pkl"
AMBIGUOUS_PROMPT_FILE = ""
CACHE_ALL_RESULTS = ""
CACHE = False

balance = []

with open(CLH_TEST_CASES_FILE, 'r', encoding='utf-8') as f:
    for i, line in tqdm(enumerate(f)):
        json_line = json.loads(line)

        labels = [token["lan"] for token in json_line["annotation"]]
        tokens = [t["token"] for t in json_line["annotation"]]
        text = json_line["text"]

        test_cases[str(i)] = {"text": text, "tokens": tokens, "labels": labels, "punct": []}


def run_ambiguous_prompt(test_cases):
    if os.path.isfile(AMBIGUOUS_PROMPT_FILE):
        with open(AMBIGUOUS_PROMPT_FILE, "rb") as f:
            prompt_results = pickle.load(f)

        missing_ids = set(test_cases.keys()) - set(prompt_results.keys())
    else:
        missing_ids = set(test_cases.keys())
        prompt_results = {}

    total_cost = 0

    if len(missing_ids) > 0:
        with tqdm(total=len(missing_ids)) as pbar:
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_id = {executor.submit(prompt_based, test_cases[id]["text"], test_cases[id]["tokens"], model="gpt-4"): id for id in missing_ids}
                for future in as_completed(future_to_id):
                    id = future_to_id[future]
                    test_labels, cost = future.result()
                    total_cost += cost
                    print(total_cost)
                    prompt_results[id] = test_labels
                    with open(AMBIGUOUS_PROMPT_FILE, "wb") as f:
                        pickle.dump(prompt_results, f)
                    pbar.update(1)

    replace_dict = {'G': 'D', 'ENGLISH': 'E', 'MIXED': 'M', '': 'D'}
    prompt_results = {k: [replace_dict.get(i, i) for i in v] for k,v in prompt_results.items()}
    
    return prompt_results

if os.path.isfile(CACHE_ALL_RESULTS):
    with open(CACHE_ALL_RESULTS, 'rb') as f:
        outputs = pickle.load(f)
else:
    outputs = {}

systems = ["lingua", "gpt", "denglisch", "eBERT", "gBERT", "mBERT", "tsBERT", "tongueswitcher"]

reset = True

# if "lingua" not in outputs:
#     outputs["lingua"] = char_baseline(test_cases)
# if "gpt" not in outputs:
#     outputs["gpt"] = run_ambiguous_prompt(test_cases)
if "denglisch" not in outputs:
    outputs["denglisch"] = denglisch_crf(test_cases)
# if "eBERT" not in outputs:
#     outputs["eBERT"] = mbert_label(test_cases, model_path=bert_model, punct=False)
# if "gBERT" not in outputs:
#     outputs["gBERT"] = mbert_label(test_cases, model_path=gbert_model, punct=False)
# if "mBERT" not in outputs:
#     outputs["mBERT"] = mbert_label(test_cases, model_path=mbert_model, punct=False)
if "tsBERT" not in outputs:
    outputs["tsBERT"] = mbert_label(test_cases, model_path=tsbert_model, punct=False)
if "tongueswitcher" not in outputs:
    outputs["tongueswitcher"] = rules_based(test_cases, flair_cache_file = FLAIR_CACHE_FILE)

if CACHE:
    with open(CACHE_ALL_RESULTS, 'wb') as f:
        pickle.dump(outputs, f)

labels = ["D", "E"]
for system in outputs.keys():
    print(f"System: {system}")
    predictions = []
    gold_labels = []
    for id in test_cases:
        idxs_to_keep = [idx for idx, l in enumerate(test_cases[id]["labels"]) if l != "X"]

        for idx in idxs_to_keep:
            predictions.extend(outputs[system][id][idx])
            gold_labels.extend(test_cases[id]["labels"][idx])

    precision, recall, f1, support = precision_recall_fscore_support(gold_labels, predictions, labels=labels, zero_division=0.0)
    for l, p, r, f, s in zip(['D', 'E', 'M', 'P'], precision, recall, f1, support):
        print(f'Class {l}: Precision={100*p:.1f}, Recall={100*r:.1f}, F1={100*f:.1f}, Support={s}')

    precision, recall, f1, support = precision_recall_fscore_support(gold_labels, predictions, average='micro', zero_division=0.0)
    print(f"{system}: P={100*precision:.1f}, R={100*recall:.1f}, F={100*f1:.1f}, Support={len(gold_labels)}")