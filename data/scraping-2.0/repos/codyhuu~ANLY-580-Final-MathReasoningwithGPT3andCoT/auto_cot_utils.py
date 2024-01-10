# -*- coding: utf-8 -*-
import openai
import numpy as np
import time
import re
import json

# function that set api key
def set_openai_apikey(api_key):
  openai.api_key = api_key

# function that takes a prompt and returns the completed sentence
def get_gpt3_response_text(prompt, engine='text-davinci-002'):
  response = openai.Completion.create(
      engine=engine,
      prompt=prompt,
      temperature=0.7,
      max_tokens=256,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0)
  if response is None or len(response) == 0:
    return 'error', ''
  return response['choices'][0]['finish_reason'], response['choices'][0]['text']

# utility functions 
# read jsonl files
def read_jsonl(path):
  with open(path) as fID:
    return [json.loads(line) for line in fID.readlines() if line]

# extract the answer from completed sentence
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def extract_answer(completion):
  match = ANS_RE.search(completion)
  if match:
    match_str = match.group(1).strip()
    match_str = match_str.replace(",", "")
    return match_str
  else:
    return INVALID_ANS

# format prompt
def format_prompt(question, keywords=False):
  if keywords:
    return 'Q: ' + question + '\n\nA: Let\'s think step by step.'
  else:
    return 'Q: ' + question + '\n\nA:'

# get data question and answer pair by index
def get_qa_by_idx(idx, data):
  return data[idx]['question'], data[idx]['answer']

# get the nearest embedding index
def get_nearest_embeddings_idx(e, embeddings):
  dist_to_embeddings = np.linalg.norm(embeddings - e, axis=1).reshape(-1,1)
  nearest_embeddings_idx = np.argmin(dist_to_embeddings, axis=0)[0]
  return nearest_embeddings_idx

def get_embedding_most_repr_label(embedding, cluster_centers):
  cluster_idx = get_nearest_embeddings_idx(embedding, cluster_centers)
  return cluster_idx

def get_embedding_nearest_idx(embedding, train_q_embeddings):
  return get_nearest_embeddings_idx(embedding, train_q_embeddings)

def extract_completion_answer(completion):
  tmp_list = []
  all_match = re.findall('[0-9.,]+', completion)
  for m in all_match:
    if m == '.':
      continue
    if m == ',':
      continue
    m = re.sub(',', '', m)
    try:
      tmp_list.append(float(m))
    except ValueError:
      continue
  if len(tmp_list) == 0:
    return None
  return tmp_list[-1]

class GPT3ArithmeticReasoning():
  def __init__(self, gpt3_engine, sbert, repr_auto_cot, most_repr_indices, train_data, train_q_embeddings, cluster_centers):
    self.gpt3_engine = gpt3_engine
    self.sbert = sbert
    self.repr_auto_cot = repr_auto_cot
    self.most_repr_indices = most_repr_indices
    self.train_q = [qa['question'] for qa in train_data]
    self.train_a = [qa['answer'] for qa in train_data]
    self.train_q_embeddings = train_q_embeddings
    self.cluster_centers = cluster_centers

  def change_gpt3_engine(self, gpt3_engine):
    self.gpt3_engine = gpt3_engine

  def generate_gpt3_prompt(self, q, prompt_method):
    e = self.sbert.encode(q)
    if prompt_method == '0-shot':
      self.prompt = format_prompt(q, keywords=False)
    elif prompt_method == '0-shot with keywords':
      self.prompt = format_prompt(q, keywords=True)
    elif prompt_method == 'Auto-COT representative question':
      label = get_embedding_most_repr_label(e, self.cluster_centers)
      self.prompt = self.repr_auto_cot[label] + format_prompt(q, keywords=True)
    elif prompt_method == 'Auto-COT nearest question':
      idx = get_embedding_nearest_idx(e, self.train_q_embeddings)
      prompt_train = format_prompt(self.train_q[idx], keywords=True)
      time.sleep(1)
      _, completion_train = get_gpt3_response_text(prompt_train, self.gpt3_engine)
      self.prompt = prompt_train + completion_train + '\n\n'+ format_prompt(q, keywords=True)
    elif prompt_method == 'Manual-COT representative question':
      label = get_embedding_most_repr_label(e, self.cluster_centers)
      idx = self.most_repr_indices[label]
      self.prompt = format_prompt(self.train_q[idx], keywords=False) + ' ' + self.train_a[idx] + '\n\n' + format_prompt(q, keywords=False)
    elif prompt_method == 'Manual-COT nearest question':
      idx = get_embedding_nearest_idx(e, self.train_q_embeddings)
      prompt_train = format_prompt(self.train_q[idx], keywords=False)
      completion_train = self.train_a[idx]
      self.prompt = prompt_train + ' ' + completion_train + '\n\n'+ format_prompt(q, keywords=False)
    else:
      print('[-] Unavailable prompt method')
      self.prompt = ''
    return self.prompt

  def get_gpt3_completion(self):
    if len(self.prompt) == 0:
      return 'error', ''
    msg, completion = get_gpt3_response_text(self.prompt, self.gpt3_engine)
    return msg, completion