# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import csv
import argparse
import numpy as np
import multiprocessing
import time
import json
import time
import random
from tqdm import tqdm
import openai
import torch
import post_process
from tenacity import (retry, stop_after_attempt, wait_random_exponential, wait_fixed)


def process_answer(answer):
  answer = answer.replace('.', '').replace(',', '').lower()
  to_be_removed = {'a', 'an', 'the', 'to', ''}
  answer_list = answer.split(' ')
  answer_list = [item for item in answer_list if item not in to_be_removed]
  return ' '.join(answer_list)


def load_anno(coco_caption_file, answer_anno_file, question_anno_file):
  if coco_caption_file is not None:
    coco_caption = json.load(open(coco_caption_file, 'r'))
    if type(coco_caption) == type({}): coco_caption = coco_caption['annotations']
  answer_anno = json.load(open(answer_anno_file, 'r'))
  question_anno = json.load(open(question_anno_file, 'r'))

  caption_dict = {}
  if coco_caption_file is not None:
    for sample in coco_caption:
      if sample['image_id'] not in caption_dict:
        caption_dict[sample['image_id']] = [sample['caption']]
      else:
        caption_dict[sample['image_id']].append(sample['caption'])
  answer_dict = {}
  for sample in answer_anno['annotations']:
    if str(sample['image_id']) + '<->' + str(sample['question_id']) not in answer_dict:
      answer_dict[str(sample['image_id']) + '<->' + str(sample['question_id'])] = [x['answer'] for x in sample['answers']]

  question_dict = {}
  for sample in question_anno['questions']:
    if str(sample['image_id']) + '<->' + str(sample['question_id']) not in question_dict:
      question_dict[str(sample['image_id']) + '<->' + str(sample['question_id'])] = sample['question']
  return caption_dict, answer_dict, question_dict


@retry(wait=wait_fixed(4), stop=stop_after_attempt(100))
def completion_with_backoff(**kwargs):
  return openai.Completion.create(**kwargs)


class PICa_RASO:

  def __init__(self, args):
    self.args = args
    self.temp = args.temp
    self.eval = args.eval
    self.tag_num = args.tag_num
    self.max_gen = 15
    _,self.answer_dict,self.question_dict = \
        load_anno(None, '%s/mscoco_val2014_annotations.json'%args.coco_path, \
            '%s/OpenEnded_mscoco_val2014_questions.json'%args.coco_path)
    self.val_keys = list(self.question_dict.keys())

    ## load cached image representation (Coco caption & Tags)
    self.inputtext_dict = self.load_cachetext()

    self.traincontext_caption_dict,self.traincontext_answer_dict,self.traincontext_question_dict = \
        load_anno('%s/captions_train2014.json'%args.coco_path, \
            '%s/mscoco_train2014_annotations.json'%args.coco_path, \
            '%s/OpenEnded_mscoco_train2014_questions.json'%args.coco_path)
    self.train_keys = list(self.traincontext_answer_dict.keys())
    self.load_similarity()
    self.gen_answers = None

  def inference(self):
    answers = []
    for key in tqdm(self.val_keys):
      answers.append(self.sample_inference(key))
    return answers

  def inference_reasons(self):
    all_reasons = []
    for i, key in tqdm(enumerate(self.val_keys)):
      img_key = int(key.split('<->')[0])
      question, answer, caption = self.question_dict[key], self.answer_dict[key], self.inputtext_dict[img_key]
      caption_i = caption[random.randint(0, len(caption) - 1)]
      reasons = []
      prompt_base = open('prompts_cot.txt', 'r').read()
      prompt_base += 'Context: %s\n===\n' % caption_i
      prompt = prompt_base + f'Question: {question}\nAnswer:'
      gen_reason = self.request_prompt(prompt, to_add_max_length=80, temp=0.7)
      reasons.append(gen_reason)
      all_reasons.append('===reasoning==='.join(reasons))
    return all_reasons

  def request_prompt(self, prompt, to_add_max_length=0, temp=0.001):
    response = completion_with_backoff(model="code-davinci-002",
                                       prompt=prompt,
                                       max_tokens=to_add_max_length,
                                       temperature=temp,
                                       stream=False,
                                       stop=["\n", "<|endoftext|>"])
    if not 'choices' in response:
      print('No response')
      return 'NO_RESPONSE'
    gen_answer = response['choices'][0]['text'][1:]
    return gen_answer

  def sample_inference(self, key):
    img_key = int(key.split('<->')[0])
    question, answer, caption = self.question_dict[key], self.answer_dict[key], self.inputtext_dict[img_key]
    caption_i = caption[random.randint(0, len(caption) - 1)]
    pred_answer_list, pred_prob_list = [], []
    context_key_list = self.get_context_keys(key, self.args.similarity_metric, self.args.n_shot * self.args.n_ensemble)
    prompt = ''
    prompt_no_context = ''
    for repeat in range(self.args.n_ensemble):
      generated = ''
      prompt = 'Please answer the question according to the above context.\n===\n'
      if self.args.nocap:
        prompt = 'Please list all the possible answers to the question.\n===\n'
      prompt_no_context = prompt
      for ni in range(self.args.n_shot):
        if context_key_list is None:
          context_key = self.train_keys[random.randint(0, len(self.train_keys) - 1)]
        else:
          context_key = context_key_list[ni + self.args.n_shot * repeat]
        img_context_key = int(context_key.split('<->')[0])
        while True:  ## make sure get context with valid question and answer
          if len(self.traincontext_question_dict[context_key]) != 0 and len(self.traincontext_answer_dict[context_key][0]) != 0:
            break
          context_key = self.train_keys[random.randint(0, len(self.train_keys) - 1)]
        prompt += 'Context: %s\n===\n' % self.traincontext_caption_dict[img_context_key][random.randint(
            0,
            len(self.traincontext_caption_dict[img_context_key]) - 1)]
        to_add = f'Q: {self.traincontext_question_dict[context_key]}\nA: {", or ".join(list(set(self.traincontext_answer_dict[context_key])))}.\n\n===\n'
        prompt += to_add
        prompt_no_context += to_add
      prompt += 'Context: %s\n===\n' % caption_i
      prompt += f'Q: {question}\nA:'
      prompt_no_context += f'Q: {question}\nA:'

      gen_answer = self.request_prompt(prompt, to_add_max_length=self.max_gen, temp=self.temp)
      generated += gen_answer
      if self.args.nocap:
        gen_answer_no_context = self.request_prompt(prompt_no_context, to_add_max_length=self.max_gen)
        generated += ', or or or, ' + gen_answer_no_context

      pred_answer_list.append(generated)
      pred_prob_list.append(1)

    pred_answer = pred_answer_list[0]
    counter = 0
    for ii in range(len(answer)):
      # eval is 'recall':
      for pred_a in pred_answer.split(', or '):
        if process_answer(pred_a) == answer[ii]: counter += 1
      else:
        if process_answer(pred_answer.split(', or ')[0]) == answer[ii]: counter += 1

    save_prompt = prompt + ', or or or, ' + prompt_no_context
    return (key, pred_answer, save_prompt, min(1., counter), ';;'.join(answer))

  def get_context_keys(self, key, metric, n):
    if metric == 'question':
      lineid = self.valkey2idx[key]
      similarity = np.matmul(self.train_feature, self.val_feature[lineid, :])
      index = similarity.argsort()[-n:][::-1]
      return [self.train_idx[str(x)] for x in index]
    elif metric == 'imagequestion':
      ## combined with Q-similairty (image+question)
      lineid = self.valkey2idx[key]
      question_similarity = np.matmul(self.train_feature, self.val_feature[lineid, :])
      ## end of Q-similairty
      similarity = question_similarity + np.matmul(self.image_train_feature, self.image_val_feature[lineid, :])
      index = similarity.argsort()[-n:][::-1]
      return [self.train_idx[str(x)] for x in index]
    else:
      return None

  def load_similarity(self):
    val_idx = json.load(open('%s/okvqa_qa_line2sample_idx_val2014.json' % self.args.similarity_path, 'r'))
    self.valkey2idx = {}
    for ii in val_idx:
      self.valkey2idx[val_idx[ii]] = int(ii)
    if self.args.similarity_metric == 'question':
      self.train_feature = np.load('%s/coco_clip_vitb16_train2014_okvqa_question.npy' % self.args.similarity_path)
      self.val_feature = np.load('%s/coco_clip_vitb16_val2014_okvqa_question.npy' % self.args.similarity_path)
      self.train_idx = json.load(open('%s/okvqa_qa_line2sample_idx_train2014.json' % self.args.similarity_path, 'r'))
    elif self.args.similarity_metric == 'imagequestion':
      self.train_feature = np.load('%s/coco_clip_vitb16_train2014_okvqa_question.npy' % self.args.similarity_path)
      self.val_feature = np.load('%s/coco_clip_vitb16_val2014_okvqa_question.npy' % self.args.similarity_path)
      self.train_idx = json.load(open('%s/okvqa_qa_line2sample_idx_train2014.json' % self.args.similarity_path, 'r'))
      self.image_train_feature = np.load('%s/coco_clip_vitb16_train2014_okvqa_convertedidx_image.npy' % self.args.similarity_path)
      self.image_val_feature = np.load('%s/coco_clip_vitb16_val2014_okvqa_convertedidx_image.npy' % self.args.similarity_path)

  def load_tags(self):
    tags_dict = {}
    tagging_pred_file = '%s/test.score.json.tsv' % self.args.tag_path
    read_tsv = csv.reader(open(tagging_pred_file, 'r'), delimiter="\t")
    for row in read_tsv:
      image_id, tags = int(row[0]), json.loads(row[1])
      tag_str = ', '.join([x['class'] for x in tags])
      tags_dict[image_id] = tag_str
    tagging_pred_file = '%s/val.score.json.tsv' % self.args.tag_path
    read_tsv = csv.reader(open(tagging_pred_file, 'r'), delimiter="\t")
    for row in read_tsv:
      image_id, tags = int(row[0]), json.loads(row[1])
      tag_str = ', '.join([x['class'] for x in tags])
      tags_dict[image_id] = tag_str
    tagging_pred_file = '%s/train.score.json.tsv' % self.args.tag_path
    read_tsv = csv.reader(open(tagging_pred_file, 'r'), delimiter="\t")
    for row in read_tsv:
      image_id, tags = int(row[0]), json.loads(row[1])
      tag_str = ', '.join([x['class'] for x in tags])
      tags_dict[image_id] = tag_str
    return tags_dict

  def load_cachetext(self):
    read_tsv = csv.reader(open(self.args.valcaption_file, 'r'), delimiter="\t")
    caption_dict = {}
    if 'tag' in self.args.caption_type:
      tags_dict = self.load_tags()
    if self.args.caption_type == 'vinvl_tag':
      for row in read_tsv:
        if self.tag_num >= 0:
          to_add = row[1].split('caption": "')[1].split('", "conf"')[0] + '. ' + tags_dict[int(row[0])][:self.tag_num]
        else:
          to_add = row[1].split('caption": "')[1].split('", "conf"')[0] + '. ' + tags_dict[int(row[0])]
        if int(row[0]) not in caption_dict:
          caption_dict[int(row[0])] = [to_add]
        else:
          caption_dict[int(row[0])].append(to_add)
    else:
      for row in read_tsv:
        if int(row[0]) not in caption_dict:
          caption_dict[int(row[0])] = [row[1].split('caption": "')[1].split('", "conf"')[0]]
        else:
          caption_dict[int(row[0])].append(row[1].split('caption": "')[1].split('", "conf"')[0])
    return caption_dict


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--apikey', type=str, default='xxxxxx', help='api key; https://openai.com/api/')
  parser.add_argument('--organization', type=str, default='xxxxxx', help='api engine; https://openai.com/api/')
  parser.add_argument('--caption_type', type=str, default='vinvl_tag', help='vinvl_tag, vinvl')
  parser.add_argument('--tag_num', type=int, default=-1, help='tag number')
  parser.add_argument('--n_shot', type=int, default=16, help="number of shots")
  parser.add_argument('--n_ensemble', type=int, default=1, help="number of ensemble")
  parser.add_argument('--similarity_metric', type=str, default='imagequestion', help="random/question/imagequestion")
  parser.add_argument('--valcaption_file', type=str, default='input_text/vinvl_caption/VinVL_base_val2014.tsv')
  parser.add_argument('--tag_path', type=str, default='input_text/coco_caption_pred_tags')
  parser.add_argument('--coco_path', type=str, default='coco_annotations')
  parser.add_argument('--similarity_path', type=str, default='coco_clip_new')
  # new parameters
  parser.add_argument('--nocap', type=int, default=0, help='also use prompt with no caption in context')
  parser.add_argument('--eval', default='recall', help='acc/recall')
  parser.add_argument('--temp', type=float, default=0.001, help='temperature')
  parser.add_argument('--output_path', type=str, default='output')
  args = parser.parse_args()
  print(args)
  openai.api_key = args.apikey
  openai.organization = args.organization

  okvqa = PICa_RASO(args)

  okvqa.device = 'cuda'

  # main inference
  answers = okvqa.inference()

  # Save result Following PiCA
  prediction = {}
  acc = 0.
  for answer in answers:
    prediction[answer[0]] = [answer[1], answer[2], answer[4]]
    acc += float(answer[3])

  format_prediction = []
  for answer in answers:
    format_prediction.append({"answer": answer[1], "gold": answer[4], "question_id": int(answer[0].split('<->')[1])})

  print(acc * 100. / len(answers), len(answers))
  acc = acc * 100. / len(answers)

  ## if save final predictions
  os.system("mkdir -p %s" % args.output_path)
  os.system("mkdir -p %s/prompt_answer" % args.output_path)
  os.system("mkdir -p %s/format_answer" % args.output_path)
  output_name = f'PICa_codex_{args.caption_type}{args.tag_num}_n{args.n_shot}_repeat{args.n_ensemble}_{args.similarity_metric}_{args.temp}{"_nocap" * bool(args.nocap)}_{args.eval}_{acc}.json'
  json.dump(prediction, open("%s/prompt_answer/%s" % (args.output_path, output_name), 'w'))
  json.dump(format_prediction, open("%s/format_answer/%s" % (args.output_path, output_name), 'w'))

  gen_answers = post_process.load_candidates(prediction)[0]
  labels = post_process.load_label(prediction)
  print(f'average number of answer:, {sum([len(d) for d in gen_answers]) / len(gen_answers)}')

  post_process.get_stat(gen_answers, labels)

  # Chain-of-thought inference
  okvqa.gen_answers = [g[:5] for g in gen_answers]
  reasons = okvqa.inference_reasons()

  output_name = f'PICa_codex_{args.caption_type}{args.tag_num}_n{args.n_shot}_repeat{args.n_ensemble}_{args.similarity_metric}_{args.temp}{"_nocap" * bool(args.nocap)}_reason.json'
  os.system("mkdir -p %s/candidate_reasoning" % args.output_path)
  json.dump(reasons, open(f'output/candidate_reasoning/{output_name}.json', 'w'))


if __name__ == '__main__':
  main()
