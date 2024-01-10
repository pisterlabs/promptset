#!/bin/python3
import os
import hashlib
import json

from args import parse_args, TYPE_SEQ2SEQ, printd
from data_processor import run_parser
from generator import OpenAIGenerator, HFGenerator
import init

def main():
	print("Loading config ...")
	args, generate_args = parse_args()
	print(json.dumps(generate_args, indent=4))
	if args.prompt_tuning:
		print(json.dumps(args.prompt_tuning, indent=4))

	# Create corpus dir if not exists
	os.makedirs(args.corpus, exist_ok=True)

	print("Loading tokenizer ...")
	tokenizer, isOpenAI, seq2seq = init.tokenizer(args)
	printd("SEQ2SEQ: " + str(seq2seq))
	print("	EOS token:", tokenizer.eos_token)

	if isOpenAI and 'OPENAI_API_KEY' not in os.environ:
		raise Exception("Please set OPENAI_API_KEY env variable")

	processor = init.processor(args, seq2seq, tokenizer, isOpenAI)

	total_max_length = tokenizer.model_max_length
	if args.type == TYPE_SEQ2SEQ and seq2seq != "codet5p": # Big codet5+ uses input in ouput (except ft)
		total_max_length = 2*tokenizer.model_max_length
	print("	Model total max tokens:", total_max_length)
	print("	Max encode tokens:", processor.max_encode_length)

	print("Loading model ...")
	model = init.model(args, isOpenAI)

	print("Parsing code ...")

	code_only = False
	if args.prompt_tuning:
		code_only = args.prompt_tuning['code_only']

	source_code = run_parser(args.parser, args.func, code_only)
	input_ids, system_ids = processor.encode(source_code)
	printd("--------------INPUT---------------") # For debugging
	printd(tokenizer.decode(input_ids))
	printd("----------------------------------")
	print("	Encoded tokens:", len(input_ids))

	decode_len = tokenizer.model_max_length-len(input_ids)
	if args.type == TYPE_SEQ2SEQ and seq2seq != "codet5p":
		decode_len = tokenizer.model_max_length

	if args.gen_length != -1 and decode_len > args.gen_length:
		decode_len = args.gen_length

	generate_args['max_new_tokens'] = decode_len
	print("	Max decode tokens:", decode_len)

	print("Generating ...")

	stop_token = processor.stop_token()
	printd("STOP TOKEN: "+json.dumps(stop_token))
	if isOpenAI:
		generator = OpenAIGenerator(model, tokenizer, stop_token, args.legacy, **generate_args)
	else:
		generator = HFGenerator(model, tokenizer, stop_token, seq2seq, **generate_args)

	total = 0
	new_seeds = 0
	for output in generator.generate(input_ids, system_ids):
		printd("--------------OUTPUT--------------")  # For debugging
		printd(output)
		printd("----------------------------------")

		seeds = processor.extract(output)

		printd("----------EXTRACTED SEEDS---------")
		for seed in seeds:
			printd(seed)
			printd("----------------------------------")

		new_seeds += save_seeds(args.corpus, seeds)

		total += len(seeds)
	
	print()
	print(f"	Generated {total} initial seed files.")
	print(f"	Total new unique seeds saved: {new_seeds}")

def save_seeds(corpus_dir: str, seeds: list[str]) -> int:
	"""
	Save each output as a file in the given directory after converting it to bytes.
	The filename is the SHA1 hash of its content.
	If a file already exists, it skips the seed.
	Raises an exception if there's an error.
	"""

	new_seeds = 0

	for seed in seeds:
		# Convert string to bytes
		seed_bytes = seed.encode('utf-8', 'surrogatepass')

		# Calculate SHA1 hash of the bytes
		sha1_hash = hashlib.sha1(seed_bytes).hexdigest()

		# Construct the full path to save the file
		file_path = os.path.join(corpus_dir, sha1_hash)

		if os.path.exists(file_path):
			continue

		try:
			# Write bytes to the file
			with open(file_path, 'wb') as file:
				file.write(seed_bytes)
			new_seeds += 1
		except Exception as e:
			raise Exception(f"Error while writing to file {file_path}: {str(e)}")

	return new_seeds


if __name__ == "__main__":
	main()
