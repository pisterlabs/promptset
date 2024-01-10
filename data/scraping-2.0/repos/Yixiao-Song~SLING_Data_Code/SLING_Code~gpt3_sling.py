import math
import numpy as np
import openai
import tqdm, glob, json
from collections import defaultdict

# Functions

def get_response(prompt: str, max_tokens=150, temperature=0.7, \
				 top_p=1, n=1, logprobs=1, stop=None, echo=True):
	response = openai.Completion.create(engine="davinci",
										prompt=prompt,
										max_tokens=max_tokens,
										temperature=temperature,
										top_p=top_p,
										n=n,
										logprobs=logprobs,
										stop=stop,
										echo=echo)
	return response


def perplexity(log_probs):
	N = len(log_probs)
	return math.exp((-1/N) * np.sum(log_probs))


def evaluate_response(response, max_tokens):
	response_dict = dict(response['choices'][0])
	text = response_dict['text']

	log_probs = response_dict['logprobs']['token_logprobs'][1:]
	ppl_prompt = perplexity(log_probs)

	return {
		'prompt_ppl': ppl_prompt,
		'text': text
	}

# Eval
# Read in SLING data
sling_files = glob.glob("SLING_Data/**/*.jsonl", recursive = True)

mp_dict_list = []
for sling_file in sling_files:
	dir = sling_file.split("/")
	phenomenon = dir[1]
	paradigm = dir[2].replace(".jsonl", "")
	good_sent, bad_sent = [], []

	with open(sling_file, "r") as file:
		mp_dict_list.extend([json.loads(x) for x in file.read().strip().split("\n")])

	for mp_dict in mp_dict_list:
		good_sent.append(mp_dict["sentence_good"])
		bad_sent.append(mp_dict["sentence_bad"])
	
	print(f"LOADED\tPHENOMENON {phenomenon}\tPARADIGM {paradigm}")
	with open("outputs/gpt3_result_sling.txt", 'a+') as file:
		file.write(f"{phenomenon}\n\t{paradigm}\n")

	correct = 0
	incorrect = 0

	for good, bad in zip(good_sent, bad_sent):
		response_good = get_response(good, max_tokens = 0)
		response_bad = get_response(bad, max_tokens = 0)
		good_ppl = evaluate_response(response_good, max_tokens=0)['prompt_ppl']
		bad_ppl = evaluate_response(response_bad, max_tokens=0)['prompt_ppl']
		if good_ppl < bad_ppl:
			correct += 1
		else:
			incorrect += 1

	assert correct + incorrect == 1000

	print(f"\t{paradigm}\t{correct/10:.4f}")
	with open("outputs/gpt3_result_sling.txt", 'a+') as file:
		file.write(f"\t{paradigm}\t{correct/10:.4f}\n")
