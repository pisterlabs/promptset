import os, urllib, json
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import torch
import tiktoken

from args import TYPE_SEQ2SEQ, TYPE_CAUSAL
from data_processor import PromptTuneProcessor, FineTuneProcessor
from generator import OpenAIGenerator, HFGenerator

def model(args, isOpenAI):
	if isOpenAI:
		return args.model


	base_name, isLoRA = get_model_base_name(args.model)
	name_or_path = args.model
	if isLoRA:
		name_or_path = base_name

	device = "cuda:0" if torch.cuda.is_available() else "cpu"

	config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)
	torch_dtype = None
	if device == "cuda:0":
		torch_dtype = torch.bfloat16 # Use half-precision bfloat16 for non-fp16 models (requires CUDA)
		if config.torch_dtype == torch.float16:
			torch_dtype = torch.float16

	if args.type == TYPE_CAUSAL:
		model = AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=torch_dtype, trust_remote_code=True, device_map=args.device_map, low_cpu_mem_usage=True)
	if args.type == TYPE_SEQ2SEQ:
		model = AutoModelForSeq2SeqLM.from_pretrained(name_or_path, torch_dtype=torch_dtype, trust_remote_code=True, device_map=args.device_map, low_cpu_mem_usage=True)
	if isLoRA:
		model = PeftModel.from_pretrained(model, args.model, torch_dtype=torch_dtype, trust_remote_code=True, device_map=args.device_map, low_cpu_mem_usage=True)
		model = model.merge_and_unload()

	return model.to(device)

def tokenizer(args):
	try:
		tokenizer = OpenAITokenizer(args.model, args.legacy)
		isOpenAI = True
		isLoRA = False
	except KeyError:
		base_name, isLoRA = get_model_base_name(args.model)
		tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
		isOpenAI = False

	seq2seq = False
	if not isOpenAI and args.type == TYPE_SEQ2SEQ:
		name_or_path = args.model
		if isLoRA:
			name_or_path = base_name
		config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)
		if config.model_type == "t5":
			tokenizer.model_max_length = 2048 # Overwrite incorrect max length for small codet5+
			tokenizer.extra_token_id = tokenizer.encode("<extra_id_0>", add_special_tokens=False)[0]
		seq2seq = config.model_type
		if not args.prompt_tuning:
			seq2seq += "-ft"

	if args.length > 0:
		tokenizer.model_max_length = args.length
	if (not hasattr(tokenizer, "model_max_length")) or tokenizer.model_max_length > 1e29:
		tokenizer.model_max_length = int(input("No default max model length. Enter max length: "))
		# TODO: Some hard-coded max lengths for known models?

	return tokenizer, isOpenAI, seq2seq

def get_model_base_name(name_or_path: str) -> str:
	config_file = os.path.join(name_or_path, "config.json")
	lora_config_file = os.path.join(name_or_path, "adapter_config.json")

	if os.path.exists(config_file):
		with open(config_file, "r") as f:
			config = json.load(f)
			return config.get('_name_or_path', name_or_path), False
	elif os.path.exists(lora_config_file):
		with open(lora_config_file, "r") as f:
			config = json.load(f)
			return config.get('base_model_name_or_path', name_or_path), True
	else:
		return name_or_path, False

def processor(args, seq2seq, tokenizer, isOpenAI):
	if args.type == TYPE_CAUSAL or seq2seq == "codet5p": # Big codet5+ uses input in ouput (except ft)
		max_encode_length = int(tokenizer.model_max_length*.75) # reserve 1/4 of model max length for generation (TODO: should depend on args.gen_length if set)
	elif args.type == TYPE_SEQ2SEQ:
		max_encode_length = tokenizer.model_max_length
	if isOpenAI and not args.legacy:
		max_encode_length -= 11 # OpenAI uses 11 extra tokens for role (system, user) input

	if args.prompt_tuning:
		processor = PromptTuneProcessor(tokenizer, seq2seq, max_encode_length, args.pt_count, **args.prompt_tuning)
	else:
		processor = FineTuneProcessor(tokenizer, seq2seq, args.split, max_encode_length)

	return processor

class OpenAITokenizer: # Wrapper class to make OpenAI tokenizer compatible
	def __init__(self, name: str, legacy: bool):
		if legacy:
			split = name.split(':', 1)
			self.enc = tiktoken.encoding_for_model(split[0])
		else:
			self.enc = tiktoken.encoding_for_model(name)

		self.eos_token = "<|endoftext|>"
		if legacy and len(split) > 1:
			self.eos_token = " END" # Legacy ft-models use " END" as eos_token (assuming OpenAI's default fine-tune format)

		eos_token_id = self.enc.encode(self.eos_token, allowed_special="all")
		if len(eos_token_id) != 1:
			raise Exception("too many tokens for eos_token_id")
		self.eos_token_id = eos_token_id[0]

	def encode(self, text, truncation=False, max_length=-1, add_special_tokens=False):
		if not truncation:
			return self.enc.encode(text)

		return self.enc.encode(text)[:max_length]

	def decode(self, tokens):
		return self.enc.decode(tokens)
