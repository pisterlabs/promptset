'''
Evaluate finetuned GPT3 on the in-domain test set of existing benchmarks: MPE, ADEPT.
End-of-prompt separator:\n\n###\n\n
Start-of-completion separator:a whitespace
End token:\n, ### or any other special token which doesn't appear within any completion.
'''
import csv
import json
import numpy as np
import openai

import os
import sys
os.chdir("../../../..")
root_dir = os.getcwd()

# config
from configuration import Config
config_path = (f'source/Qa/finetune_on_benchmark/gpt3/config.json')
config = Config.from_json_file(config_path)

sys.path.append(f"{root_dir}/source")
from Qa.utils import compute_scores, n_classes_dict_benchmark, task_fieldnames, query


# name of our finetuned models for each RNPC task
finetuned_model_dict = {
	"MPTE":{
		"ada":"ada:ft-ccb-lab-members-2021-08-12-01-09-11",
		"curie":"curie:ft-ccb-lab-members-2021-08-12-02-24-03"
	},
	"EPC":{
		"ada":"ada:ft-ccb-lab-members-2021-08-13-00-40-39",
		"curie":"curie:ft-ccb-lab-members-2021-08-13-02-39-17",
	}
}


if __name__ == "__main__":

	# config
	RNPC_task = eval(config.task)

	# not supporting the corresponding benchmarks for SPTE, since the finetuned GPT3 models on SNLI/MNLI are not publicly available
	if RNPC_task not in ["MPTE", "EPC"]:
		raise ValueError(f"Unspported task: {RNPC_task}. Please choose from 'MPTE', and 'EPC'.")

	benchmark_dict = {"MPTE": "MPE",
	                  "EPC": "ADEPT"}
	benchmark = benchmark_dict[RNPC_task]

	model = eval(config.gpt3_model)
	trial = config.trial

	# dirs
	jsonl_dir = f"data/existing_benchmarks/{benchmark}/jsonl"
	output_dir = f"output_dir/benchmarks/{benchmark}"
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)
	if trial:
		frn = f"{jsonl_dir}/test_sample.jsonl"
	else:
		frn = f"{jsonl_dir}/test.jsonl"
	fwn = f"{output_dir}/gpt3-{model}_test.tsv"


	gold_labels, pred_labels = [], []

	# For SPTE, the finetuning datasets (MNLI, SNLI) are too large.
	# OpenAI doesn't allow datasets of such sizes at the time of the work.

	# Evaluate on MPE
	if benchmark == "MPE":
		with open(frn, 'r') as fr, open(fwn, 'w') as fw:
			writer = csv.DictWriter(fw, delimiter="\t", fieldnames=task_fieldnames[RNPC_task])
			writer.writeheader()

			for row_id, row_obj in enumerate(fr.readlines()):
				if row_id % 50 == 0:
					print(f"{row_id} examples finished.")

				row = json.loads(row_obj)
				prompt, completion = row["prompt"], row["completion"]
				gold_label = int(completion.strip())

				pred_label, label_probs = query(finetuned_model_dict[RNPC_task][model], prompt, n_classes_dict_benchmark[RNPC_task])
				pred_prob = label_probs[pred_label] / sum(label_probs.values())

				gold_labels.append(gold_label)
				pred_labels.append(pred_label)

				cols = prompt.split("\n")
				premises, hypothesis = cols[0], cols[1]
				premises = premises[len("Premises: "):]
				hypothesis = hypothesis[len("Hypothesis: "):]

				new_row = {}
				new_row["premise"] = premises
				new_row["hypothesis"] = hypothesis
				new_row["gold label"] = gold_label
				new_row["pred label"] = pred_label
				new_row["confidence"] = pred_prob
				writer.writerow(new_row)

	# Evaluate on ADEPT
	elif benchmark == "ADEPT":

		with open(frn, 'r') as fr, open(fwn, 'a') as fw:
			writer = csv.DictWriter(fw, delimiter="\t", fieldnames=task_fieldnames[RNPC_task])
			writer.writeheader()

			for row_id, row_obj in enumerate(fr.readlines()):

				if row_id % 50 == 0:
					print(f"{row_id} examples finished.")

				row = json.loads(row_obj)
				prompt, completion = row["prompt"], row["completion"]
				gold_label = int(completion.strip())

				pred_label, label_probs = query(finetuned_model_dict[RNPC_task][model], prompt, n_classes_dict_benchmark[RNPC_task])
				pred_prob = label_probs[pred_label] / sum(label_probs.values())

				gold_labels.append(gold_label)
				pred_labels.append(pred_label)

				cols = prompt.split("\n")
				event1, event2 = cols[0], cols[1]
				event1 = event1[len("Event 1: "):]
				event2 = event2[len("Event 2: "):]

				new_row = {}
				new_row["first_event"] = event1
				new_row["second_event"] = event2
				new_row["gold label"] = gold_label
				new_row["pred label"] = pred_label
				new_row["confidence"] = pred_prob
				writer.writerow(new_row)

	scores =compute_scores(n_classes_dict_benchmark[RNPC_task], gold_labels, pred_labels)
	print(scores)