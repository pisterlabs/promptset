# this file implements the gpt as a service, based on a pretrained model from the source
# https://github.com/huggingface/pytorch-pretrained-BERT

import torch
from pytorch_transformers import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

# Tokenized input
text = "Who was Jim Henson ? Jim Henson was a puppeteer"
tokenized_text = tokenizer.tokenize(text)

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])

print(tokens_tensor.size())
# Load pre-trained model (weights)
model = OpenAIGPTModel.from_pretrained('openai-gpt')
model.eval()

cuda = torch.device('cuda:1')

# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to(cuda)
model.to(cuda)

# Predict hidden states features for each layer
with torch.no_grad():
    hidden_states = model(tokens_tensor)

print(len(hidden_states))

for i, state in enumerate(hidden_states):
    print("Layer {}'s Shape: {}".format(i, state.size()))
    
