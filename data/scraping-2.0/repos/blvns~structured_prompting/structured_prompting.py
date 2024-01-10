#!/usr/bin/env python
# -*- coding: utf-8 -*- 

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import torch.nn.functional as F
from tqdm import tqdm
import subprocess

import os
import sys
import random
import argparse
import numpy as np
import pickle

from pathlib import Path

#Uncomment to use openai models
#import openai
#Fill in with your token to use OpenAI model
#openai.api_key = None
import time #sleeping to handle rate limits
import copy

from utils import *

#Fill in with where you want the results to be stored
RESULTS_PATH = None
#Fill in with where you save the task example indices
INDICES_PATH = None

DELIMS = { 'slash': '/', 'colon': ':', 'dash': '-', 'underscore': '_', 'equals': '=', 'space': ' '}
BIO_STRUCTURE ={
	'bio': ['O', 'B', 'I']
}

PLACEHOLDERS = {
	'pos': ['11', '12', '13', '14', '15', '16', '17', '18', '19',\
		'20', '21', '22', '23', '24', '25', '26', '27'],
	'ner': ['O', 'B-11', 'I-11', 'B-12', 'I-12', 'B-13', 'I-13',\
		'B-14', 'I-14'],
	'chunk': ['O', 'B-11', 'I-11', 'B-12', 'I-12', 'B-13', 'I-13',\
		'B-14', 'I-14', 'B-15', 'I-15', 'B-16', 'I-16', 'B-17', 'I-17',\
		'B-18', 'I-18', 'B-19', 'I-19', 'B-20', 'I-20', 'B-21', 'I-21']
}

LABEL_NAMES = {
	'pos': ['adjective', 'adverb', 'interjection', 'noun', 'proper noun', 'verb',\
	'preposition', 'auxiliary verb', 'coordinating conjunction', 'determiner', 'number', 'participle',\
	'pronoun', 'subordinating conjunction', 'punctuation', 'symbol', 'other'],
	'ner': ['O', 'B-person', 'I-person', 'B-organization', 'I-organization', 'B-location',\
	'I-location', 'B-miscellaneous', 'I-miscellaneous']
}

def _load_indices(task, split, eval_ids=True):
	if eval_ids:
		index_path = INDICES_PATH+'{}-{}-eval-indices.txt'.format(task, split)
	else:
		index_path = INDICES_PATH+'{}-{}-prompt-indices.txt'.format(task, split)

	print(index_path)
	all_ids = []
	curr_ids = []
	with open(index_path, 'r') as f:
		for line in f:
			if len(line.strip()) == 0:
				all_ids.append(curr_ids)
				curr_ids = []
			else:
				idx = int(line.strip())
				curr_ids.append(idx)
	if len(curr_ids) > 0:
		all_ids.append(curr_ids)
	return all_ids

def construct_in_context_data(examples, indices, bos_token, k=1, tags_only=False):
	prompt_example_template = 'Context: {}\nTagged: {}\n'
	#if args.task_prefix: in_context = '{} {}\n'.format(bos_token, TASK_NAME[args.task])
	#else: in_context = '{} '.format(bos_token)
	in_context = '{} '.format(bos_token)
	l = 0
	for idx in indices[:k]:
		text, labels = examples[idx]

		c = ' '.join(text)	
		if tags_only: t = ' '.join(labels)
		else: t = ' '.join(['{}{}{}'.format(x, args.tag_delimiter, y) for x, y in zip(text, labels)])
		e = prompt_example_template.format(c, t)
		in_context += e

		l += len(text)

	return in_context

#a derangement with no overlaps with original labels
def shuffle_labels(prompt_data, eval_data, original_labels, task):
	#keep randomly shuffling until new labels are all different from original
	if task == 'pos':
		overlap_flag = True
		while overlap_flag:
			shuffled_labels = original_labels.copy()
			random.shuffle(shuffled_labels)
			overlap_flag = False
			for x, y in zip(original_labels, shuffled_labels):
				if x == y: 
					overlap_flag = True
					break
	else:
		if task == 'ner': base_labels = ['PER', 'ORG', 'LOC', 'MISC']
		#chunking...
		else: base_labels = ['ADJP', 'ADVP', 'CONJP', 'INTJ', 'LST',\
		'NP', 'PP', 'PRT', 'SBAR', 'UCP', 'VP']
		overlap_flag = True
		while overlap_flag:
			shuffled_labels = base_labels.copy()
			random.shuffle(shuffled_labels)
			overlap_flag = False
			for x, y in zip(base_labels, shuffled_labels):
				if x == y: 
					overlap_flag = True
					break

		#recreate bio labels with shuffled set
		bio_labels = ['O']
		for s in shuffled_labels:
			bio_labels.append('B-{}'.format(s))
			bio_labels.append('I-{}'.format(s))
		shuffled_labels = bio_labels

	shuffled_prompt_data = []
	for sentence, labels in prompt_data:
		l_arr = []
		for l in labels:
			if l == '_':
				l_arr.append(l)
			else:
				l_idx = original_labels.index(l)
				new_l = shuffled_labels[l_idx]
				l_arr.append(new_l)
		shuffled_prompt_data.append((sentence, l_arr))

	shuffled_eval_data = []
	for sentence, labels in eval_data:
		l_arr = []
		for l in labels:
			if l == '_':
				l_arr.append(l)
			else:
				l_idx = original_labels.index(l)
				new_l = shuffled_labels[l_idx]
				l_arr.append(new_l)
		shuffled_eval_data.append((sentence, l_arr))

	return shuffled_prompt_data, shuffled_eval_data, shuffled_labels

#replace surface form of labels with placeholders with no semantic ties to task
def placeholder_labels(prompt_data, eval_data, original_labels, task):
	placeholder_labels = PLACEHOLDERS[task]
	print(placeholder_labels)

	alt_prompt_data = []
	for sentence, labels in prompt_data:
		l_arr = []
		for l in labels:
			if l == '_':
				l_arr.append(l)
			else:
				l_idx = original_labels.index(l)
				new_l = placeholder_labels[l_idx]
				l_arr.append(new_l)
		alt_prompt_data.append((sentence, l_arr))

	alt_eval_data = []
	for sentence, labels in eval_data:
		l_arr = []
		for l in labels:
			if l == '_':
				l_arr.append(l)
			else:
				l_idx = original_labels.index(l)
				new_l = placeholder_labels[l_idx]
				l_arr.append(new_l)
		alt_eval_data.append((sentence, l_arr))

	return alt_prompt_data, alt_eval_data, placeholder_labels

#replace surface form of labels with plain english word for part-of-speech
def words_as_labels(prompt_data, eval_data, original_labels, task):
	alt_labels = LABEL_NAMES[task]
	print(alt_labels)

	alt_prompt_data = []
	for sentence, labels in prompt_data:
		l_arr = []
		for l in labels:
			if l == '_':
				l_arr.append(l)
			else:
				l_idx = original_labels.index(l)
				new_l = alt_labels[l_idx]
				l_arr.append(new_l)
		alt_prompt_data.append((sentence, l_arr))

	alt_eval_data = []
	for sentence, labels in eval_data:
		l_arr = []
		for l in labels:
			if l == '_':
				l_arr.append(l)
			else:
				l_idx = original_labels.index(l)
				new_l = alt_labels[l_idx]
				l_arr.append(new_l)
		alt_eval_data.append((sentence, l_arr))

	return alt_prompt_data, alt_eval_data, alt_labels

#reverting bio tags from alt format ablation experiments
def revert_bio_tags(preds, golds):
	reverted_preds = []
	reverted_golds = []

	for sent in preds:
		revert_sent = []
		for l in sent:
			if '-' in l:
				s, c = l.split('-')
				s_idx = BIO_STRUCTURE['alt'].index(s)
				s2 = BIO_STRUCTURE['bio'][s_idx]
				revert_sent.append('{}-{}'.format(s2, c))
			else:
				l_idx = BIO_STRUCTURE['alt'].index(l)
				revert_sent.append(BIO_STRUCTURE['bio'][l_idx])
		reverted_preds.append(revert_sent)

	for sent in golds:
		revert_sent = []
		for l in sent:
			if '-' in l:
				s, c = l.split('-')
				s_idx = BIO_STRUCTURE['alt'].index(s)
				s2 = BIO_STRUCTURE['bio'][s_idx]
				revert_sent.append('{}-{}'.format(s2, c))
			else:
				l_idx = BIO_STRUCTURE['alt'].index(l)
				revert_sent.append(BIO_STRUCTURE['bio'][l_idx])
		reverted_golds.append(revert_sent)
	return reverted_preds, reverted_golds

def _bio_constraints(label_space, prev_label):
	bio_stuct = BIO_STRUCTURE['bio'] 
	constrained_space = []
	for l, x in label_space:
		#can always predict O or B- tags
		if l[0] == bio_stuct[0] or l[0] == bio_stuct[1]:
			constrained_space.append((l, x))

		#can also predict I-X (where X is in previous label)
		elif prev_label and prev_label[0] != bio_stuct[0]:
			prev_label_content = prev_label.split('-')[1]
			l_content = l.split('-')[1]
			if l[0] == bio_stuct[2] and l_content == prev_label_content:
				constrained_space.append((l, x))

	return constrained_space

def _score_seq_example_gpt_neo(text, labels, label_space, model, tokenizer, prompt_cache, task, tags_only=False):
	model_inputs = ''
	gold_labels = []
	pred_labels = []

	#add context of eval example to in context training data
	example_prompt = 'Context: {}\nTagged:'.format(' '.join(text))

	input_ids = tokenizer(example_prompt, add_special_tokens=False, return_tensors='pt')['input_ids']
	with torch.no_grad():
		#input_ids = input_ids.to("cuda:0")
		input_ids = input_ids.to("cuda")
		output = model(input_ids, past_key_values=prompt_cache, use_cache=True)
	#track model state after seeing context for this example
	example_cache = output['past_key_values']

	model_inputs += example_prompt

	#iteratively add words and get predicted labels for example
	input_string = ''
	prev_label = None
	for word, gold in zip(text, labels):
		if tags_only: word_input = ' '
		else: word_input = ' {}{}'.format(word, args.tag_delimiter)
		input_string += word_input

		model_inputs += input_string

		#pass next word through model
		input_ids = tokenizer(input_string, add_special_tokens=False, return_tensors='pt')['input_ids']

		with torch.no_grad():
			#input_ids = input_ids.to("cuda:0")
			input_ids = input_ids.to("cuda")
			output = model(input_ids, past_key_values=example_cache, use_cache=True)

		#track model state after seeing next word and before we generate labels
		base_cache = output['past_key_values']
		logits = output['logits'].squeeze()
		if logits.dim() > 1: logits = logits[-1]
		base_probs = F.log_softmax(logits, dim=-1)

		scores = []

		#enforce BIO constraints for appropriate tasks
		if task in ['ner', 'chunk']:
			constrained_space = _bio_constraints(label_space, prev_label)
		else:
			constrained_space = label_space

		for candidate_label, candidate_ids in constrained_space:

			#copy of probs, label_scores for each label
			l_probs = torch.clone(base_probs)
			l_cache = tuple(tuple(torch.clone(c2) for c2 in c1) for c1 in base_cache)
			l_scores = []
			for i, cid in enumerate(candidate_ids):
				l_scores.append(l_probs[cid].item())

				#keep running through if not last id for label
				if i+1 < len(candidate_ids):
					x = torch.LongTensor([[cid]])
					#x = x.to("cuda:0")
					x = x.to("cuda")
					output = model(x, past_key_values=l_cache, use_cache=True)
					#tracking so we don't have to run whole input through each time
					l_cache = output['past_key_values']
					logits = output['logits'].squeeze()
					l_probs = F.log_softmax(logits, dim=-1)

			avg_lscore = sum(l_scores)/len(l_scores)
			scores.append((candidate_label, avg_lscore))

		scores = sorted(scores, key=lambda x: x[1], reverse=True)
		pred = scores[0][0]
		gold_labels.append(gold)
		pred_labels.append(pred)
		
		#add pred label to context
		if tags_only: input_string = ' '+pred
		else: input_string = pred
		example_cache = base_cache
		prev_label = pred

	#show final predictions per example
	model_inputs += input_string
	#print(model_inputs)
	#input('...')

	return gold_labels, pred_labels

def _openai_request(model, prompt, unconstrained=False):
	r = None
	while r == None:
		try:
			if unconstrained:
				r = openai.Completion.create(model=model, prompt=prompt, max_tokens=256, stop='\n', temperature=0, logprobs=1)
			else:
				r = openai.Completion.create(model=model, prompt=prompt, max_tokens=1, temperature=0, logprobs=100)
			#sleeping to handle rate limits for codex
			time.sleep(1)
		except openai.error.RateLimitError:
			print('Rate limit timed out')
			time.sleep(30)
		except openai.error.ServiceUnavailableError:
			print('Server Unavailable')
			time.sleep(30)
		except openai.error.APIError:
			print("Misc. API error")
			time.sleep(30)
	if unconstrained:
		scores = r['choices'][0]['text']
		cost = r['usage']['total_tokens']
	else:
		scores = r['choices'][0]['logprobs']['top_logprobs'][0]
		cost = r['usage']['total_tokens']
	return scores, cost

def _score_seq_example_openai_unconstrained(text, base_prompt, labels, label_space, model, task):
	token_cost = 0
	gold_labels = labels
	pred_labels = []
	model_input = base_prompt

	#add context of eval example to in context training data
	example_prompt = 'Context: {}\nTagged:'.format(' '.join(text))
	model_input += example_prompt

	#start generation by giving "Tagged: <first_word><delim>"
	first_word = text[0]
	word_input = ' {}{}'.format(first_word, args.tag_delimiter)
	model_input += word_input

	gen_text, cost = _openai_request(model, model_input, unconstrained=True)
	token_cost += cost

	#score to output against labels, and get cost from openai response
	#(predict noun for any missing words?)
	outputs = gen_text.split(' ')
	outputs[0] = '{}{}{}'.format(first_word, args.tag_delimiter, outputs[0])
	outputs = [o.split('_') for o in outputs]

	j = 0
	for i in range(0, len(text)):
		g = text[i].replace("'", '’')
		gw = outputs[j][0].replace("'", '’')

		#keep checking if word generated here doesn't match
		while j < len(outputs) and g != gw:
			j += 1 
			if j >= len(outputs): break
			gw = outputs[j][0].replace("'", '’')
			
		if j >= len(outputs): break

		if len(outputs[j]) > 1:
			pred = outputs[j][1]
			pred_labels.append(pred)

	#handling for missing tags (i.e., null placeholders)
	while len(pred_labels) < len(gold_labels):
		pred_labels.append('O') #not in POS space and output labeled spans in BIO

	return gold_labels, pred_labels, token_cost

#WARNING: This is very expensive due to not caching context between API calls! 
def _score_seq_example_openai_constrained(text, base_prompt, labels, label_space, model, task):
	token_cost = 0
	gold_labels = []
	pred_labels = []
	model_input = base_prompt

	#add context of eval example to in context training data
	example_prompt = 'Context: {}\nTagged:'.format(' '.join(text))
	model_input += example_prompt

	#iteratively add words and get predicted labels for example
	#https://help.openai.com/en/articles/5955598-is-api-usage-subject-to-any-rate-limits
	prev_label = None
	for word, gold in zip(text, labels):

		word_input = ' {}{}'.format(word, args.tag_delimiter) #slash format
		model_input += word_input

		r, t = _openai_request(model, model_input, unconstrained=False)
		token_cost += t
		base_probs = {}
		for _, l in label_space:
			l = l[0]
			if l in r: base_probs[l] = r[l]
			else: base_probs[l] = float('-inf')

		scores = []

		#enforce BIO constraints for appropriate tasks
		if prev_label and task in ['ner', 'chunk']:
			constrained_space = _bio_constraints(label_space, prev_label)
		else:
			constrained_space = label_space

		for candidate_label, candidate_tokens in constrained_space:

			#copy of probs, label_scores for each label
			l_probs = copy.deepcopy(base_probs)
			l_scores = []
			l_input = model_input
			for i, ctoken in enumerate(candidate_tokens):
				l_scores.append(l_probs[ctoken])

				#keep running through if not last id for label
				if i+1 < len(candidate_tokens):
					l_input += ctoken
					r, t = _openai_request(model, l_input, unconstrained=False)
					token_cost += t
					l_probs = {}
					next_tok = candidate_tokens[i+1]
					if next_tok in r: 
						l_probs[next_tok] = r[next_tok]
					else: 
						l_probs[next_tok] = float('-inf')
						break

			if float('-inf') in l_scores: avg_lscore = float('-inf')
			else: avg_lscore = sum(l_scores)/len(l_scores)
			scores.append((candidate_label, avg_lscore))

		inf_check = [1 for _, s in scores if s == float('-inf')]
		if len(inf_check) == len(scores): 
			pred = 'NOUN' #predict most common tag if all have no prob
		else:
			scores = sorted(scores, key=lambda x: x[1], reverse=True)
			pred = scores[0][0]

		gold_labels.append(gold)
		pred_labels.append(pred)
		
		#add pred label to context
		model_input += pred
		prev_label = pred

	return gold_labels, pred_labels, token_cost

def score_seq_openai(eval_data, label_space, base_prompt, model, tokenizer, task='pos', unconstrained=False):
	label_space = [(x, [tokenizer.decode(z, add_special_tokens=False) for z in y]) for x, y in label_space]

	gold_labels = []
	pred_labels = []

	token_cost = 0
	model_cost = 0.02 if model == 'davinci' else 0.002 #curie

	#get prompt cache
	logged_index = -1
	f = Path('{}-tmp-log.pkl'.format(model))
	if f.exists():
		with open('{}-tmp-log.pkl'.format(model), 'rb') as f:
			x = pickle.load(f)
		logged_index, token_cost, gold_labels, pred_labels = x

	example_index = 0
	for text, labels in tqdm(eval_data):

		example_index += 1
		if example_index <= logged_index:
			continue
		
		if unconstrained:
			g, p, t = _score_seq_example_openai_unconstrained(text, base_prompt, labels, label_space, model, task=task)
		else:
			g, p, t = _score_seq_example_openai_constrained(text, base_prompt, labels, label_space, model, task=task)

		gold_labels.append(g)
		pred_labels.append(p)

		token_cost += t

		if example_index%20 == 0:
			print('tokens submitted = '+str(token_cost))
			print('running cost = ${}'.format((token_cost/1000)*model_cost))
			sys.stdout.flush()
			tmp_log = (example_index, token_cost, gold_labels, pred_labels)
			
			with open('{}-tmp-log.pkl'.format(model), 'wb') as f:
				pickle.dump(tmp_log, f)



	print('total tokens submitted = '+str(token_cost))
	print('total cost = ${}'.format((token_cost/1000)*model_cost))
	return pred_labels, gold_labels

def score_seq(eval_data, label_space, base_prompt, model, tokenizer, task='pos', no_past=False, oracle=False, tags_only=False):
	gold_labels = []
	pred_labels = []

	#get prompt cache
	prompt_cache = None
	if len(base_prompt) > 0:
		input_ids = tokenizer(base_prompt, add_special_tokens=False, return_tensors='pt')['input_ids']

		with torch.no_grad():
			input_ids = input_ids.to("cuda")
			output = model(input_ids, use_cache=True)
		#track model state after k-shot examples
		#(so we don't run them through every time)
		prompt_cache = output['past_key_values']

	for text, labels in tqdm(eval_data):
		g, p = _score_seq_example_gpt_neo(text, labels, label_space, model, tokenizer, prompt_cache, task=task, tags_only=tags_only)

		gold_labels.append(g)
		pred_labels.append(p)

	return pred_labels, gold_labels

def preprocess_data(args):
	if args.task == 'pos':
		return pos_data(args.eval_on_test)
	elif args.task == 'chunk':
		return chunking_data()
	elif args.task == 'ner':
		return ner_data(args.eval_on_test)

def run_experiment(args, tokenizer, model, model_type):
	k = args.k

	eval_split = 'dev'
	#no dev set for chunking dataset
	if args.eval_on_test or args.task == 'chunk':
		eval_split = 'test'

	prompt_data, eval_data, labels = preprocess_data(args)

	if args.shuffle_labels:
		prompt_data, eval_data, labels = shuffle_labels(prompt_data, eval_data, labels, task=args.task)
	elif args.placeholder_labels:
		prompt_data, eval_data, labels = placeholder_labels(prompt_data, eval_data, labels, task=args.task)
	elif args.words_as_labels and args.task in ['pos', 'ner']:
		prompt_data, eval_data, labels = words_as_labels(prompt_data, eval_data, labels, task=args.task)

	if args.tags_only or args.tag_delimiter == ' ':
		label_ids = [tokenizer(' '+l, add_special_tokens=False)['input_ids'] for l in labels]
	else:
		label_ids = [tokenizer(l, add_special_tokens=False)['input_ids'] for l in labels]
	labels = list(zip(labels, label_ids))
	
	scores_arr = []
	eval_indexes = _load_indices(args.task, eval_split)
	prompt_indexes = _load_indices(args.task, eval_split, eval_ids=False)

	#use the same eval data for every run (only varying the demonstrations)
	eval_data_subset = [eval_data[i] for i in eval_indexes[0]]

	for run_id in range(args.num_runs):

		#only do the run specified by run_index if given
		if args.run_index > -1 and run_id != args.run_index:
			continue

		base_prompt = construct_in_context_data(prompt_data, prompt_indexes[run_id], tokenizer.bos_token, k=k, tags_only=args.tags_only)
		print(base_prompt)

		if model_type == 'gpt3':
			preds, golds = score_seq_openai(eval_data_subset, labels, base_prompt, model, tokenizer, task=args.task, unconstrained=args.unconstrained)
		else:
			preds, golds = score_seq(eval_data_subset, labels, base_prompt, model, tokenizer, task=args.task, tags_only=args.tags_only, model_type=model_type)

		#write predictions, gold labels to file
		output_txt = ''
		for g, p in zip(golds, preds):
			output_txt += '\n'.join(['{} {}'.format(x, y) for x, y in zip(g, p)])+'\n\n'

		m = args.model_name.split('/')[1]
		if args.shuffle_labels: 
			out_path = '{}/{}_{}_{}_k{}_shuffle_{}_results.txt'.format(RESULTS_PATH, args.task, m, eval_split, args.k, run_id)
		elif args.placeholder_labels: 
			out_path = '{}/{}_{}_{}_k{}_placeholder_{}_results.txt'.format(RESULTS_PATH, args.task, m, eval_split, args.k, run_id)
		elif args.words_as_labels:
			out_path = '{}/{}_{}_{}_k{}_words_{}_results.txt'.format(RESULTS_PATH, args.task, m, eval_split, args.k, run_id)
		else:	
			out_path = '{}/{}_{}_{}_k{}_{}_results.txt'.format(RESULTS_PATH, args.task, m, eval_split, args.k, run_id)
		with open(out_path, 'w') as f: f.write(output_txt)

		#calculate F1 if appropriate
		if args.task in ['ner', 'chunk']:
			eval_cmd = 'perl eval/conlleval.pl < {}'.format(out_path)
			output = subprocess.Popen(eval_cmd, stdout=subprocess.PIPE, shell=True, stderr=subprocess.DEVNULL).communicate()[0]
			output = [x.decode("utf-8") for x in output.splitlines()]	
			score = float(output[1].split('FB1:')[1].strip())
		else:
			score = ud_pos_accuracy(golds, preds)

		print(score)
		scores_arr.append(score)
		print(scores_arr)

def main(args):
	if 'EleutherAI' in args.model_name: model_type = 'gpt-neo'
	else: model_type = 'gpt3' 

	if model_type == 'gpt-neo':
		tokenizer = AutoTokenizer.from_pretrained("{}".format(args.model_name))
	else:
		tokenizer = AutoTokenizer.from_pretrained("{}".format('EleutherAI/gpt-neo-125M'))
	
	#see https://github.com/huggingface/transformers/issues/15642
	if args.model_name == 'EleutherAI/gpt-neox-20b':
		model = init_gpt_neox(fp16=True)
		model = model.eval()
	elif args.model_name in ['facebook/opt-13b', 'facebook/opt-30b', 'facebook/opt-66b']:
		model = init_opt(args.model_name, fp16=True)
		model = model.eval()
	elif model_type == 'gpt3':
		model = args.model_name.split('/')[1]
	#note: need 48G GPU to run > 2.7B models
	#and custom loading for larger models that don't fit on single GPU (> ~6.7B parameters)
	else:
		model = AutoModelForCausalLM.from_pretrained("{}".format(args.model_name))
		model = model.to("cuda")
		model = model.eval()

	args.tag_delimiter = DELIMS[args.tag_delimiter]
	run_experiment(args, tokenizer, model, model_type)
		
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
	    "--rand_seed", default=42, type=int
	)
	parser.add_argument(
	    "--k", default=1, type=int
	)
	parser.add_argument(
	    "--num_runs", default=5, type=int
	)
	#overrides num_runs to only do a single, specific run of m runs
	parser.add_argument(
	    "--run_index", default=-1, type=int
	)
	parser.add_argument(
	    "--task", default='pos', type=str,
	    choices=['pos', 'chunk', 'ner']
	)
	parser.add_argument(
	    "--model_name", default='gpt-neo-125M', type=str,
	    choices=['EleutherAI/gpt-neo-125M', 'EleutherAI/gpt-neo-1.3B',\
	    'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-j-6B', 'EleutherAI/gpt-neox-20b',\
	    'openai/curie', 'openai/davinci']
	)
	#generate tag sequence with greedy decoding directly from model
	#with no formatting constraints enforced
	parser.add_argument(
	    "--unconstrained", action='store_true'
	)
	#run on test instead of development data
	parser.add_argument(
	    "--eval_on_test", action='store_true'
	)
	parser.add_argument(
	    "--shuffle_labels", action='store_true'
	)
	parser.add_argument(
	    "--placeholder_labels", action='store_true'
	)
	parser.add_argument(
	    "--words_as_labels", action='store_true'
	)
	parser.add_argument(
	    "--tags_only", action='store_true'
	)
	parser.add_argument(
	    "--tag_delimiter", type=str, default='underscore',
	    choices=['slash', 'colon', 'dash', 'underscore', 'equals', 'space']
	)


	args = parser.parse_args()
	print(args)

	torch.manual_seed(args.rand_seed)
	os.environ['PYTHONHASHSEED'] = str(args.rand_seed)
	torch.cuda.manual_seed(args.rand_seed)
	torch.cuda.manual_seed_all(args.rand_seed)   
	np.random.seed(args.rand_seed)
	random.seed(args.rand_seed)
	torch.backends.cudnn.benchmark=False
	torch.backends.cudnn.deterministic=True

	main(args)

#EOF