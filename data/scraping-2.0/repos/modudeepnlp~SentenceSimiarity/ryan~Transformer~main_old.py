"""
OpenGPT Documentation: https://huggingface.co/pytorch-transformers/model_doc/gpt.html
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable

from tqdm import tqdm, trange

import random
import numpy as np
import logging
import json
# import matplotlib.pyplot as plt

from models.transformer import TransformerModel, DoubleHeadModel
from models.loss import ClassificationLossCompute
# from models.opt import OpenAIAdam

from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils.data import Data
from utils import configs

import preprocessing.custom_dataset as custom_dataset
from torch.utils.data import DataLoader

import config as config
import pickle

from pytorch_transformers import (OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer, OpenAIGPTConfig, OpenAIGPTModel,
                                  AdamW, cached_path, WEIGHTS_NAME, CONFIG_NAME)
from pytorch_transformers.modeling_openai import OpenAIGPTPreTrainedModel
from pytorch_transformers.modeling_utils import SequenceSummary

config_path = 'config/configs.transformer.json'
args = configs.Config.load(config_path)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {}, n_gpu {}".format(device, n_gpu))

if not os.path.exists(args.data_out):
	os.makedirs(args.data_out)

print('Building Model')

def load_pick(file_nm):
	with open(file_nm, 'rb') as f:
		label, df = pickle.load(f)
		print("load compeleted")
		return label, df

dev_label, dev_df = load_pick('data_in/dev.pkl')
test_label, test_df = load_pick('data_in/test.pkl')
train_label, train_df = load_pick('data_in/train.pkl')

# Load tokenizer and model
# This loading functions also add new tokens and embeddings called `special tokens`
# These new embeddings will be fine-tuned on the RocStories dataset
special_tokens = ['_start_', '_delimiter_', '_classify_']
tokenizer = OpenAIGPTTokenizer.from_pretrained(args.model_name, special_tokens=special_tokens)
special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
model = OpenAIGPTDoubleHeadsModel.from_pretrained(args.model_name, num_special_tokens=len(special_tokens))


special_tokens = ['<bos>', '<del>', '<eos>', '<pad>']
tokenizer = OpenAIGPTTokenizer.from_pretrained(args.model_name, special_tokens=special_tokens)  # OpenAI용 토크나이저 불러오기
tokenizer.add_tokens(special_tokens)

config = OpenAIGPTConfig.from_pretrained('openai-gpt')
config.num_labels = 3
config.vocab_size = len(tokenizer)
config.summary_type = 'last'

tokenizer.bos_token = '<bos>'
tokenizer.eos_token = '<eos>'
tokenizer.sep_token = '<del>'
tokenizer.pad_token = '<pad>'

class CustomClassifier(OpenAIGPTPreTrainedModel):

	def __init__(self, config):
		super(CustomClassifier, self).__init__(config)

		self.transformer = OpenAIGPTModel(config)
		self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
		self.apply(self.init_weights)
		self.multiple_choice_head = SequenceSummary(config)
		self.tie_weights()

	def tie_weights(self):
		""" Make sure we are sharing the input and output embeddings.
			Export to TorchScript can't handle parameter sharing so we are cloning them instead.
		"""
		self._tie_or_clone_weights(self.lm_head,
		                           self.transformer.tokens_embed)

	def forward(self, input_ids, mc_token_ids=None, lm_labels=None, mc_labels=None, token_type_ids=None,
	            position_ids=None, head_mask=None):
		transformer_outputs = self.transformer(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
		                                       head_mask=head_mask)

		hidden_states = transformer_outputs[0]
		lm_logits = self.lm_head(hidden_states)
		# mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)
		mc_logits = self.multiple_choice_head(hidden_states)

		outputs = (lm_logits, mc_logits) + transformer_outputs[1:]

		return mc_logits


# model = CustomClassifier.from_pretrained(args.model_name, num_special_tokens=special_tokens)
model = CustomClassifier(config)
model.to(device)

# model.resize_token_embeddings(len(tokenizer))

def build_tensor(label, sentence, device, batch_size):
	torch_label = torch.tensor(label, dtype=torch.long).to(device)
	torch_sent = torch.tensor(sentence, dtype=torch.long).to(device)
	dataset = torch.utils.data.TensorDataset(torch_label, torch_sent)
	data_loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True)
	return data_loader

dev_loader = build_tensor(dev_label, dev_df, device, args.batch_size)
test_loader = build_tensor(test_label, test_df, device, args.batch_size)
train_loader = build_tensor(train_label, train_df, device, args.batch_size)

loss_fn = torch.nn.CrossEntropyLoss()

num_total_steps = 100000
num_warmup_steps = 10000
warmup_propotion = float(num_warmup_steps) / float(num_total_steps)

optimizer = AdamW(model.parameters(), lr=args.learning_rate, correct_bias=False)
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)  # PyTorch scheduler

def eval(loader):
	total = 0
	correct = 0
	for n, (eval_label, eval_entail_sent) in enumerate(loader):

		eval_outputs = model(eval_entail_sent)
		_, pred = torch.max(eval_outputs.data, 1)
		total += eval_label.size(0)
		correct += (pred == label).sum()

	acc = 100 * (correct.cpu().numpy()/total)
	return acc


step_list = []
loss_list = []
acc_test_list = []
acc_dev_list = []

train_loss = 0
step = 0
for epoch in range(args.epoch):

	# Training Phase
	for n, (label, entailment_sent) in enumerate(train_loader):

		model.train()

		optimizer.zero_grad()
		outputs = model(entailment_sent)
		loss = loss_fn(outputs, label)
		loss.backward()
		scheduler.step()
		optimizer.step()

		train_loss += loss.item()

		step += 1

		if n % 50 == 0:

			epoch_loss = train_loss / 50
			print("epoch {}, loss: {}".format(n, epoch_loss))
			train_loss = 0

		if n % 500 == 0:

			model.eval()
			step_list.append(step)
			loss_list.append(epoch_loss)
			acc_test = eval(test_loader)
			acc_dev = eval(dev_loader)
			acc_test_list.append(acc_test)
			acc_dev_list.append(acc_dev)

			print(acc_test_list)
			print(acc_dev_list)