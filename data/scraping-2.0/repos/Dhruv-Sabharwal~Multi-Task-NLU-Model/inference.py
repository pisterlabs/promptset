import math
import time
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#!pip install transformers
#!pip install ftfy
#!pip install spacy
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, AdamW
%load_ext tensorboard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/BaselineModel')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt').to(device)
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

SPECIAL_TOKENS = ["<bos>", "<eos>", "<system>", "<user>", "<slots>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<system>', '<user>', '<slots>']}
MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

add_special_tokens_(model, tokenizer)

model.load_state_dict(torch.load("C:/Users/dhruv/Desktop/ASP sem 1/Capstone/Conversational model/saved_models/final_with_nlu/model_18.pt"), strict=False)
model.eval()

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

# Here is how to use this function for top-p sampling
temperature = 1    # between 0 and 2  (default = 1)
top_k = 0           # between 0 and 200 (default = 0)
top_p = 0.8          # between 0 and 1  (default = 0.9)


def get_pred_data(input_sentence):
  pred_input_ids = tokenizer(input_sentence)['input_ids']
  pred_token_type_ids = []
  for i in range(len(pred_input_ids)):
    pred_tokens = []
    if(i%2==0):
      for j in range(len(pred_input_ids[i])):
        pred_tokens.append('<system>')
    else:
      for j in range(len(pred_input_ids[i])):
        pred_tokens.append('<user>')
    pred_token_type_ids.append(pred_tokens)

  testing_input_ids = []
  testing_token_type_ids = []
  for i in range(len(pred_input_ids)):
    for j in range(len(pred_input_ids[i])):
      testing_input_ids.append(pred_input_ids[i][j])
      testing_token_type_ids.append(tokenizer(pred_token_type_ids[i][j])['input_ids'][0])

  testing_input_ids = torch.tensor(testing_input_ids, dtype = torch.long, device=device).unsqueeze(0)
  testing_token_type_ids = torch.tensor(testing_token_type_ids, dtype = torch.long, device=device).unsqueeze(0)
  return testing_input_ids, testing_token_type_ids


def get_next_token(testing_input_ids, testing_token_type_ids):
  # Get logits with a forward pass in our model (input is pre-defined)
  logits = model(input_ids=testing_input_ids, token_type_ids=testing_token_type_ids)

  if isinstance(logits, tuple):  # for gpt2 and maybe others
    logits = logits[0]

  # Keep only the last token predictions of the first batch item (batch size 1), apply a temperature coefficient and filter
  logits = logits[0, -1, :] / temperature
  filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

  # Sample from the filtered distribution
  probabilities = F.softmax(filtered_logits, dim=-1)
  next_token = torch.multinomial(probabilities, 1)
  return tokenizer.decode(next_token, skip_special_tokens=False)

input_sentence = ["<bos> <system> Hello , welcome to the automated restaurant system . How may I help you ?"]
conversation_history = ["Hello , welcome to the automated restaurant system . How may I help you ?"]
while(True):
  print(conversation_history[-1])
  user_sentence = input()
  input_sentence.append("<user> " + user_sentence)
  input_sentence.append("<slots>")
  conversation_history.append(user_sentence)

  while(True):
    testing_input_ids, testing_token_type_ids = get_pred_data(input_sentence)
    next_token = get_next_token(testing_input_ids, testing_token_type_ids)
    if(next_token=='<eos>'):
      break
    input_sentence[-1] += ' '+next_token
  
  if(input_sentence[-1][:]=="<slots> <system> have a nice day ."):
    print(input_sentence[-1][:])
    break
  conversation_history.append(input_sentence[-1][:])
