import os
import sys
#import ipdb; ipdb.set_trace() #debug
import torch
import tqdm
import datetime
import argparse
import pandas as pd
from glob import glob
from itertools import islice
from logzero import logger
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel


def extract_ith_contexts(indexed_tokens, ith, max_context_len):
	len_list = [len(tokenized_sent) for tokenized_sent in indexed_tokens]
	target_sent_len = len_list[ith - 1]

	full_context_before_len = sum(len_list[:ith - 1])
	full_context_after_len = sum(len_list[ith:])

	tmp_context_before_len = (max_context_len - target_sent_len) * 0.5
	tmp_context_after_len = (max_context_len - target_sent_len) * 0.5

	if sum(len_list) <= max_context_len:  # full_context_before_len <= context_before_len and full_context_after_len <= context_after_len
		max_context_before_len, max_context_after_len = full_context_before_len, full_context_after_len
	elif full_context_before_len <= tmp_context_before_len and full_context_after_len > tmp_context_after_len:
		max_context_before_len = full_context_before_len
		max_context_after_len = max_context_len - (full_context_before_len + target_sent_len)
	elif full_context_before_len > tmp_context_before_len and full_context_after_len <= tmp_context_after_len:
		max_context_after_len = full_context_after_len
		max_context_before_len = max_context_len - (full_context_after_len + target_sent_len)
	elif full_context_before_len > tmp_context_before_len and full_context_after_len > tmp_context_after_len:
		max_context_before_len = tmp_context_before_len
		max_context_after_len = tmp_context_after_len
	else:
		print("you wrote wrong if statement")
		sys.exit(1)

	# extract context before
	context_before = []
	context_before_len = []
	for tokenized_sent, sent_len in zip(indexed_tokens[:ith - 1][::-1], len_list[:ith - 1][::-1]):
		if sum(context_before_len) + sent_len <= max_context_before_len:
			context_before = context_before + tokenized_sent[::-1]
			context_before_len.append(sent_len)

	# extract context after
	context_after = []
	context_after_len = []
	for tokenized_sent, sent_len in zip(indexed_tokens[ith:], len_list[ith:]):
		if sum(context_after_len) + sent_len <= max_context_after_len:
			context_after = context_after + tokenized_sent
			context_after_len.append(sent_len)

	context_before.reverse()

	#print("Processing {}th sentence".format(ith))
	#print("full context len: {}, {}".format(full_context_before_len, full_context_after_len))
	#print("max context len: {}, {}".format(max_context_before_len, max_context_after_len))
	#print("context len: {}, {}".format(len(context_before), len(context_after)))
	#print()
	return context_before, context_after


def compute_prob_for_rest_of_story(model, device, indexed_tokens, ith, max_context_len):

	context_before, context_after = extract_ith_contexts(indexed_tokens, ith, max_context_len)
	context_after_len = len(context_after)

	# include target sentence
	input_tensor_w_sk = torch.tensor(context_before + indexed_tokens[ith-1] + context_after).unsqueeze(0)
	# does not include target sentence
	input_tensor_wo_sk = torch.tensor(context_before + context_after).unsqueeze(0)

	if device >= 0:
		device_name = 'cuda:{}'.format(device)
		input_tensor_w_sk = input_tensor_w_sk.to(device_name)
		input_tensor_wo_sk = input_tensor_wo_sk.to(device_name)
	else:
		pass

	with torch.no_grad():
		outputs = model(input_tensor_w_sk)

		logits, *_ = outputs

		# Shift so that tokens < n predict n
		shift_logits = logits[..., :-1, :].contiguous()
		shift_labels = input_tensor_w_sk[..., 1:].contiguous()

		loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
		loss_w_sk = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

	with torch.no_grad():
		outputs = model(input_tensor_wo_sk)
		logits, *_ = outputs

		# Shift so that tokens < n predict n
		shift_logits = logits[..., :-1, :].contiguous()
		shift_labels = input_tensor_wo_sk[..., 1:].contiguous()

		loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
		loss_wo_sk = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

	unnormalized_prob_w_sk = loss_w_sk[-context_after_len:].sum().item()
	unnormalized_prob_wo_sk = loss_wo_sk[-context_after_len:].sum().item()

	normalized_prob_w_sk = loss_w_sk[-context_after_len:].sum().item() / context_after_len
	normalized_prob_wo_sk = loss_wo_sk[-context_after_len:].sum().item() / context_after_len

	if args.normalization == "normalize":
		return normalized_prob_w_sk, normalized_prob_wo_sk
	elif args.normalization == "unnormalize":
		return unnormalized_prob_w_sk, unnormalized_prob_wo_sk
	else:
		print("invalid normalization method")
		sys.exit(1)


def check_file_alignment(file_path_list_1, file_path_list_2):
	for file_path_1, file_path_2 in zip(file_path_list_1, file_path_list_2):
		if os.path.basename(file_path_1) == os.path.basename(file_path_1):
			#logger.info("{} is aligned {}".format(file_path_1, file_path_2))
			pass
		else:
			logger.info("{} is misaligned {}".format(file_path_1, file_path_2))
			sys.exit(1)
	return None


def main(args):

	logger.info('Preparing file path...')

	original_file_path_list = glob(os.path.normpath(args.input) + "/*")

	dt_now = datetime.datetime.now()

	new_dir_path = os.path.normpath(args.output) + "/" + str(dt_now).replace(":", "-") + "_SD_" + os.path.basename(args.model) + "_" + str(args.contextlen) + "_" + args.normalization + "_gpu_context_eot"

	os.makedirs(new_dir_path)

	output_file_path_list = [new_dir_path + "/result_" + os.path.basename(args.model) + "_" + args.normalization + "_" + os.path.basename(path) for path in original_file_path_list]

	logger.info('Loading model and tokenizer...')

	if args.model == "gpt2":
		# Load pre-trained model tokenizer (vocabulary)
		tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
		# Load pre-trained model (weights)
		model = GPT2LMHeadModel.from_pretrained('gpt2')
	elif args.model == "gpt2-medium":
		# Load pre-trained model tokenizer (vocabulary)
		tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
		# Load pre-trained model (weights)
		model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
	elif args.model =="gpt2-large":
		# Load pre-trained model tokenizer (vocabulary)
		tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
		# Load pre-trained model (weights)
		model = GPT2LMHeadModel.from_pretrained('gpt2-large')
	elif args.model == "gpt2-xl":
		# Load pre-trained model tokenizer (vocabulary)
		tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
		# Load pre-trained model (weights)
		model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
	elif args.model == "gpt":
		# Load pre-trained model tokenizer (vocabulary)
		tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
		# Load pre-trained model (weights)
		model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
	elif args.model == "distilgpt2":
		tokenizer = GPT2Tokenizer.from_pretrained('gpt2') # distilgpt2 uses GTP-2 tokenizer
		model = GPT2LMHeadModel.from_pretrained('distilgpt2')
	elif args.model == "transformer-xl":
		tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
		model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
	else:
		tokenizer = GPT2Tokenizer.from_pretrained(args.model)
		model = GPT2LMHeadModel.from_pretrained(args.model)
		logger.info("Load finetuned model from: {}".format(args.model))

	# send to cuda if necessary
	if args.gpu >= 0:
		logger.info('Using GPU #{}'.format(args.gpu))
		device_name = 'cuda:{}'.format(args.gpu)
		print(device_name)
		model.to(device_name)

	# Set the model in evaluation mode to deactivate the DropOut modules
	# This is IMPORTANT to have reproducible results during evaluation!
	model.eval()

	max_context_len = args.contextlen

	progress_bar1 = tqdm.tqdm(total=len(original_file_path_list), desc="Iteration over folktales")

	for input_file_path, output_file_path in zip(original_file_path_list, output_file_path_list):

		with open(input_file_path, encoding='utf-8') as infile:
			sentences = [line.strip().split("\t")[1] for line in infile]
			#print(sentences)
		indexed_tokens = [tokenizer.encode(sentence) for sentence in sentences]
		assert indexed_tokens[-1] == tokenizer.encode("<|endoftext|>")

		with open(input_file_path, encoding='utf-8') as infile, open(output_file_path, "w") as outfile:
			progress_bar2 = tqdm.tqdm(total=len(indexed_tokens), desc="Iteration over sentences")

			result_list = []
			for ith, line in enumerate(infile.readlines()[:-1]):
				#with_sk, without_sk = compute_prob_for_rest_of_story(model, args.gpu, indexed_tokens, ith + 1, max_context_len)

				if ith == len(sentences) - 2:  # last sentence
					if len(sentences) == 2:
						with_sk, without_sk = compute_prob_for_rest_of_story(model, args.gpu, indexed_tokens, ith, max_context_len)
					else:
						with_sk, without_sk = compute_prob_for_rest_of_story(model, args.gpu, indexed_tokens, ith + 1, max_context_len)
				else:
					with_sk, without_sk = compute_prob_for_rest_of_story(model, args.gpu, indexed_tokens[:-1], ith + 1, max_context_len)

				result_list.append([line.strip().split("\t")[0], line.strip().split("\t")[1], with_sk, without_sk, without_sk - with_sk])

				progress_bar2.update(1)

			df = pd.DataFrame(result_list, columns=["label", "sentence", "total_loss_with_sk", "total_loss_without_sk", "score"])
			df.to_csv(output_file_path, sep='\t')
			progress_bar1.update(1)
                
			print("DONE: " + output_file_path)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--event_rem_method', '-e', type=str, help='event removal method')
	parser.add_argument('--model', '-m', type=str, help='language model')
	parser.add_argument('--gpu', '-g', type=int, default=-1, help='use device')
	parser.add_argument('--normalization', '-n', type=str, help='normalization method')
	parser.add_argument('--contextlen', '-c', type=int, help='max context length')
	parser.add_argument('--input', '-i', type=str, help='directory path for inputfiles')
	parser.add_argument('--output', '-o',  help='directory path for outputfiles')

	args = parser.parse_args()
	main(args)
