#  transformer_chatbot
#  Copyright (C) 2018 Golovanov, Tselousov
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import json
import math
import os
import pickle
import random

import fasttext
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.tokenization_gpt2 import GPT2Tokenizer
from transformers.tokenization_openai import OpenAIGPTTokenizer

from model.seq2seq_vocab import Seq2seqTokenizer
from .postprocessing import augment_replica

SPECIAL_TOKENS = ['.', ',', '?', '!', ':']

class FacebookDataset(Dataset):
    @staticmethod
    def parse_data(path):
        last_index, partner_persona = -1, False
        with open(path, 'r', encoding='utf-8') as file:
            data = []
            for line in file.readlines():
                line = line.strip()

                if len(line) == 0:
                    continue

                space_idx = line.find(' ')
                if space_idx == -1:
                    dialog_idx = int(line)
                else:
                    dialog_idx = int(line[:space_idx])

                if int(dialog_idx) == 1:
                    data.append({'persona_info': [], 'dialog': [], 'candidates': []})

                dialog_line = line[space_idx + 1:].split('\t')
                dialog_line = [l.strip() for l in dialog_line]

                if dialog_line[0].startswith('your persona:'):
                    persona_info = dialog_line[0].replace('your persona: ', '')
                    if persona_info[-1] == '.' and persona_info[-2] != ' ':
                        persona_info = persona_info[:-1] + ' .'
                    data[-1]['persona_info'].append(persona_info)
                if dialog_line[0].startswith('partner\'s person'):
                    if not data[-1].__contains__('partner_persona_info'):
                        data[-1]['partner_persona_info'] = []
                    persona_info = dialog_line[0].replace('partner\'s persona: ', '')
                    if persona_info[-1] == '.' and persona_info[-2] != ' ':
                        persona_info = persona_info[:-1] + ' .'
                    data[-1]['partner_persona_info'].append(persona_info)

                elif len(dialog_line) > 1:
                    data[-1]['dialog'].append(dialog_line[0])
                    data[-1]['dialog'].append(dialog_line[1])
                if len(dialog_line) == 4:
                    data[-1]['candidates'].append(dialog_line[3].split('|')[:-1])  # the last candidate is a duplicate of the good answer (dialog_line[1])

            return data

    @staticmethod
    def parse_data_emoji(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = []
            for line in f.readlines():
                line = line.strip()
                items = line.split('\t')
                data.append({'persona_info': [], 'dialog': [], 'candidates': []})
                data[-1]['persona_info'].append(items[0])
                data[-1]['dialog'].append(items[1])
                data[-1]['dialog'].append(items[2])
            return data

    @staticmethod
    def parse_data_daily(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = []
            for line in f.readlines():
                line = line.strip()
                items = line.split('\t')
                data.append({'persona_info': [], 'dialog': [], 'candidates': []})
                data[-1]['persona_info'].append(items[0])
                for i in range(1, len(items)):
                    data[-1]['dialog'].append(items[i])
            return data

    @staticmethod
    def parse_data_weibo(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = []
            for line in f.readlines():
                line = line.strip()
                items = line.split('\t')
                data.append({'persona_info': [], 'dialog': [], 'candidates': []})
                data[-1]['dialog'].append(items[0])
                data[-1]['dialog'].append(items[1])
            return data

    @staticmethod
    def parse_data_prototype(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = []
            for line in f.readlines():
                line = line.strip()
                items = line.split('\t')
                data.append({'persona_info': [], 'dialog': [], 'candidates': []})
                data[-1]['dialog'].append(items[0])
                data[-1]['dialog'].append(items[1])
            return data

    @staticmethod
    def parse_data_entailment(path):
        with open(path, 'r', encoding='utf-8') as f:
            data =[]
            list = json.load(f)
            for item in list:
                data.append({'persona_info': [], 'dialog': [], 'candidates': []})
                if item[0][-1] == '.' and item[0][-2] != ' ':
                    item[0] = item[0][:-1] + ' .'
                data[-1]['persona_info'].append(item[0])
                data[-1]['dialog'].extend(item[1])
                data[-1]['dialog'].append(item[2])
        return data

    @staticmethod
    def parse_data_attention(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = []
            list = json.load(f)
            for item in list:
                data.append({'persona_info': item[0], 'dialog': item[1], 'candidates': []})
            return data

    @staticmethod
    def make_dataset(data, vocab, only_final=False):
        dataset = []
        if isinstance(vocab, OpenAIGPTTokenizer) or isinstance(vocab, GPT2Tokenizer) or isinstance(vocab, Seq2seqTokenizer):
            for chat in tqdm(data):
                persona_info = [vocab.encode(vocab.tokenize(s)) for s in chat['persona_info']]

                dialog = []
                if only_final:
                    for utterance in chat['dialog']:
                        dialog.append(vocab.encode(vocab.tokenize(utterance)))
                    dataset.append((persona_info, dialog[:], []))
                else:
                    for i, replica in enumerate(chat['dialog'], 1):
                        dialog.append(vocab.encode(vocab.tokenize(replica)))
                        if not i % 2:
                            if chat['candidates']:
                                candidates_ids = [vocab.encode(vocab.tokenize(c)) for c in chat['candidates'][(i - 1) // 2]]
                                dataset.append((persona_info, dialog[:], candidates_ids))
                            else:
                                dataset.append((persona_info, dialog[:], []))
                if chat.__contains__('partner_persona_info'):
                    persona_info = [vocab.encode(vocab.tokenize(s)) for s in chat['partner_persona_info']]
                    dialog = []
                    for i, replica in enumerate(chat['dialog'], 1):
                        dialog.append(vocab.encode(vocab.tokenize(replica)))
                        if i % 2 and i > 2:
                            dataset.append((persona_info, dialog[:], []))
        else:
            for chat in tqdm(data):
                persona_info = [vocab.string2ids(s) for s in chat['persona_info']]

                dialog = []
                for i, replica in enumerate(chat['dialog'], 1):
                    dialog.append(vocab.string2ids(replica))
                    if not i % 2:
                        if chat['candidates']:
                            candidates_ids = [vocab.string2ids(c) for c in chat['candidates'][(i-1)//2]]
                            dataset.append((persona_info, dialog[:], candidates_ids))
                        else:
                            dataset.append((persona_info, dialog[:], []))

        return dataset

    @staticmethod
    def make_proto_dataset(data, vocab):
        dataset = []
        if isinstance(vocab, OpenAIGPTTokenizer) or isinstance(vocab, GPT2Tokenizer) or isinstance(vocab,
                                                                                                   Seq2seqTokenizer):
            for chat in tqdm(data):
                query = vocab.encode(vocab.tokenize(chat['dialog'][0]))
                response = vocab.encode(vocab.tokenize(chat['dialog'][1]))
                insert_tokens = set(response) - set(query)
                delete_tokens = set(query) - set(response)
                dataset.append(([list(insert_tokens), list(delete_tokens)], [query, response], []))
        return dataset

    def __init__(self, paths, vocab, *, max_lengths=512,  max_y_length=80, min_infos=2, dialog_embeddings=False,
                 use_start_end=True, negative_samples=0, limit_size=-1,
                 cache=None, augment=False, aug_syn_proba=0.1, aug_vary_length=True, max_history_size=-1,
                 single_input=False, data_type='persona', parsed_data=None, few_shot=False, task_map_path=None,
                 extra_train_path=None, extra_data_type='persona', ignore_sample_indices=None, extra_cvae_utterances_path=None):
        assert min_infos > 0

        if isinstance(paths, str):
            paths = [paths]

        self.augment = augment
        self.aug_syn_proba = aug_syn_proba
        self.aug_vary_length = aug_vary_length

        self.vocab = vocab
        self.max_lengths = max_lengths
        self.max_y_length = max_y_length
        self.min_infos = min_infos
        self.dialog_embeddings = dialog_embeddings
        self.use_start_end = use_start_end
        self.negative_samples = negative_samples  # -1 => include all candidates in data instance
        self.max_history_size = max_history_size
        self.single_input = single_input
        self.data_type = data_type

        if cache and os.path.exists(cache):
            self.data = torch.load(cache)
        else:
            self.data = self._parse_data(paths, vocab, data_type, parsed_data)
            if extra_train_path is not None:
                extra_data = self._parse_data([extra_train_path], vocab, extra_data_type, None)
                self.data.extend(extra_data)
            if extra_cvae_utterances_path is not None:
                with open(extra_cvae_utterances_path, 'r') as f:
                    cvae_utterances = json.load(f)
                self._extend_cvae_utterances(cvae_utterances)
            if cache:
                torch.save(self.data, cache)

        if limit_size > 0:
            self.data = self.data[:limit_size]
        if ignore_sample_indices:
            with open(ignore_sample_indices, 'r') as f:
                ignore_indices = set(json.load(f))
            filter_data = []
            for i, d in enumerate(self.data):
                if not ignore_indices.__contains__(i):
                    filter_data.append(d)
            self.data = filter_data
        if few_shot and task_map_path is not None:
            with open(task_map_path, 'r') as f:
                self.task_map = json.load(f)
        # if mixup:
        #     self.data = self.data + self.data

    def __len__(self):
        return len(self.data)

    def _augment(self, sentences, info=False):

        if not self.augment:
            return sentences

        if info:
            n_info_samples = max(self.min_infos, random.randint(1, len(sentences)))
            n_info_samples = min(n_info_samples, len(sentences))
            sentences = random.sample(sentences, n_info_samples)
            random.shuffle(sentences)
        else:
            if self.aug_vary_length:
                begin = random.randrange(0, len(sentences) - 1, 2)
                end = random.randrange(begin + 2, len(sentences) + 1, 2)

                sentences = sentences[begin:end]

        def _try2augment(sent):
            if random.uniform(0, 1) < self.aug_syn_proba:
                sent = self.vocab.ids2string(sent)
                sent = augment_replica(sent)
                sent = self.vocab.string2ids(sent)
            return sent

        sentences = list(map(_try2augment, sentences)) if self.aug_syn_proba > 0 else sentences

        return sentences

    def _get_distractors(self, candidates):
        if self.negative_samples == 0:
            return []
        if self.negative_samples == -1:  # => include all candidates in data instance
            return candidates
        if len(candidates) >= self.negative_samples:
            distractors = random.sample(candidates, k=self.negative_samples)
        else:  # not enought candidates, sample from train dataset instead (we may sample the gold y but quite unlikely)
            distractors = random.sample(range(len(self.data)), k=self.negative_samples)
            distractors = [self.data[ids][1][-1] for ids in distractors]
        return distractors

    def _extend_cvae_utterances(self, cvae_utterances):
        for i, utterance_pair in enumerate(cvae_utterances):
            if isinstance(self.vocab, OpenAIGPTTokenizer) or isinstance(self.vocab, GPT2Tokenizer) or \
                isinstance(self.vocab, Seq2seqTokenizer):
                q = self.vocab.encode(self.vocab.tokenize(utterance_pair[0]))
                r = self.vocab.encode(self.vocab.tokenize(utterance_pair[1]))
            else:
                q = self.vocab.string2ids(utterance_pair[0])
                r = self.vocab.string2ids(utterance_pair[1])
            cur_data = (self.data[i][0], self.data[i][1][:-2] + [q, r], self.data[i][2])
            self.data.append(cur_data)

    def _parse_data(self, paths, vocab, data_type, parsed_data):
        data = None
        if data_type == 'persona':
            if not parsed_data:
                parsed_data = sum([FacebookDataset.parse_data(path) for path in paths], [])
            data = FacebookDataset.make_dataset(parsed_data, vocab)
        if data_type == 'persona_attention':
            if not parsed_data:
                parsed_data = sum([FacebookDataset.parse_data_attention(path) for path in paths], [])
            data = FacebookDataset.make_dataset(parsed_data, vocab, only_final=True)
        if data_type == 'entailment':
            if not parsed_data:
                parsed_data = sum([FacebookDataset.parse_data_entailment(path) for path in paths], [])
            data = FacebookDataset.make_dataset(parsed_data, vocab, only_final=True)
        elif data_type == 'emoji':
            if not parsed_data:
                parsed_data = sum([FacebookDataset.parse_data_emoji(path) for path in paths], [])
            data = FacebookDataset.make_dataset(parsed_data, vocab)
        elif data_type == 'daily':
            if not parsed_data:
                parsed_data = sum([FacebookDataset.parse_data_daily(path) for path in paths], [])
            data = FacebookDataset.make_dataset(parsed_data, vocab)
        elif data_type == 'prototype':
            if not parsed_data:
                parsed_data = sum([FacebookDataset.parse_data_prototype(path) for path in paths], [])
            data = FacebookDataset.make_proto_dataset(parsed_data, vocab)
        return data

    def get_tasks_dataset(self):
        tasks = []
        for k, v in self.task_map.items():
            tasks.append((k, v['ids']))
        return TaskDataset(tasks)

    def __getitem__(self, idx):
        persona_info, dialog, candidates = self.data[idx]

        if len(persona_info):
            persona_info = self._augment(persona_info, info=True)
            persona_info = sum(persona_info, [])
            if self.single_input:
                persona_info = [self.vocab.bos_id] + persona_info
                if self.dialog_embeddings:
                    persona_info = [[tok, self.vocab.talker1_bos_id] for tok in persona_info]
            elif not self.single_input and not self.dialog_embeddings:
                persona_info = [self.vocab.bos_id] + persona_info[:self.max_lengths-2]
            else:
                persona_info = [self.vocab.info_bos_id] + persona_info[:self.max_lengths-2] + \
                               [self.vocab.info_eos_id] if self.use_start_end else persona_info[:self.max_lengths]
                if self.dialog_embeddings:
                    persona_info = [[tok, self.vocab.info_dialog_id] for tok in persona_info]

        dialog = self._augment(dialog)
        candidates = self._get_distractors(candidates)

        h = []
        history_start = 0
        if self.max_history_size != -1:
            history_start = -1 - self.max_history_size
        dialog_history = dialog[history_start: -1]
        if self.single_input:
            for i, ids in enumerate(dialog_history):
                if (len(dialog_history) - i) % 2 == 0:
                    ids = [self.vocab.talker1_bos_id] + ids
                else:
                    ids = [self.vocab.talker2_bos_id] + ids
                if self.dialog_embeddings:
                    ids = [[tok, self.vocab.talker1_bos_id if (len(dialog_history) - i) % 2 == 0
                            else self.vocab.talker2_bos_id] for tok in ids]
                h.extend(ids)
        elif not self.single_input and not self.dialog_embeddings:
            for i, ids in enumerate(dialog_history):
                if (len(dialog_history) - i) % 2 == 0:
                    ids = [self.vocab.talker1_bos_id] + ids
                else:
                    ids = [self.vocab.talker2_bos_id] + ids
                h.extend(ids)
        else:
            for i, ids in enumerate(dialog_history):
                if (len(dialog_history) - i) % 2 == 0 and self.use_start_end:
                    ids = [self.vocab.talker1_bos_id] + ids + [self.vocab.talker1_eos_id]
                elif self.use_start_end:
                    ids = [self.vocab.talker2_bos_id] + ids + [self.vocab.talker2_eos_id]
                if self.dialog_embeddings:
                    ids = [[tok, self.vocab.talker1_dialog_id if (len(dialog_history) - i) % 2 == 0
                            else self.vocab.talker2_dialog_id] for tok in ids]
                h.extend(ids)
            h = h[-self.max_lengths:]

        sentences = []
        for y in (dialog[-1:] + candidates):
            if self.single_input:
                y = [self.vocab.talker1_bos_id] + y + [self.vocab.eos_id]
                if self.dialog_embeddings:
                    y = [[tok, self.vocab.talker1_bos_id] for tok in y]
                sentences.append(y)
            elif not self.single_input and not self.dialog_embeddings:
                y = [self.vocab.talker1_bos_id] + y + [self.vocab.eos_id]
                sentences.append(y)
            else:
                y = [self.vocab.bos_id] + y + [self.vocab.eos_id]
                if self.dialog_embeddings:
                    y = [[tok, self.vocab.sent_dialog_id] for tok in y]
                sentences.append(y)

        return persona_info, h, sentences[0], sentences[1:]

class TaskDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, idx):
        return self.data_list[idx]

    def __len__(self):
        return len(self.data_list)

class MixUpDataset():
    def __init__(self, path, tokenizer, fasttext_path, cache=None, th=0.4, max_neighbors=5, data_type='persona'):
        if cache and os.path.exists(cache):
            with open(cache, 'rb') as f:
                self.neighbor_dict = pickle.load(f)
        else:
            self.tokenizer = tokenizer
            self.fasttext_model = fasttext.load_model(fasttext_path)
            if data_type == 'persona':
                self.parsed_data = FacebookDataset.parse_data(path)
            elif data_type == 'emoji':
                self.parsed_data = FacebookDataset.parse_data_emoji(path)
            elif data_type == 'daily':
                self.parsed_data = FacebookDataset.parse_data_daily(path)
            token_set = set()
            for persona in tqdm(self.parsed_data):
                for k, v in persona.items():
                    for sentence in v:
                        if isinstance(sentence, list):
                            sentence = ' '.join(sentence)
                        tokens = sentence.split(' ')
                        for t in tokens:
                            if t not in SPECIAL_TOKENS:
                                if t[-1] in ['.', ',', '!', '?', ';']:
                                    token_set.add(t[:-1])
                                elif t.find('.') != -1:
                                    ts = t.split('.')
                                    for tmp_t in ts:
                                        token_set.add(tmp_t)
                                else:
                                    token_set.add(t)
            self.neighbor_dict = {}
            for token in tqdm(list(token_set), desc='Building token neighbor dict'):
                neighbors = self.fasttext_model.get_nearest_neighbors(token)
                if isinstance(self.tokenizer, Seq2seqTokenizer):
                    token_neighbors = []
                    for n in neighbors:
                        if n[0] < th:
                            break
                        encoded_n = self.tokenizer.encode([n[1]])
                        if len(encoded_n) > 0:
                            token_neighbors.append((encoded_n, n[0]))
                        if len(token_neighbors) >= max_neighbors:
                            break
                    if len(token_neighbors) != 0:
                        encoded_token = self.tokenizer.encode([token])
                        self.neighbor_dict[tuple(x for x in encoded_token)] = token_neighbors
                else:
                    ## there should be two forms for each token: blank before a token or no blank before a token
                    ## correspondingly, there should be two set of encoded neighbors
                    token_neighbors, blank_token_neighbors = [], []
                    for n in neighbors:
                        if n[0] < th:
                            break
                        encoded_n = self.tokenizer.encode(n[1])
                        token_neighbors.append((encoded_n, n[0]))
                        blank_encoded_n = self.tokenizer.encode(' ' + n[1])
                        blank_token_neighbors.append((blank_encoded_n, n[0]))
                        if len(token_neighbors) >= max_neighbors:
                            break
                    if len(token_neighbors) != 0:
                        encoded_token = self.tokenizer.encode(token)
                        blank_encoded_token = self.tokenizer.encode(' ' + token)
                        self.neighbor_dict[tuple(x for x in encoded_token)] = token_neighbors
                        self.neighbor_dict[tuple(x for x in blank_encoded_token)] = blank_token_neighbors
                # token_neighbors = []
                # if len(encoded_token) == 1:
                #     neighbors = self.fasttext_model.get_nearest_neighbors(token)
                #     for n in neighbors:
                #         encoded_n = self.tokenizer.encode(n[1])
                #         if len(encoded_n) == 1:
                #             token_neighbors.append((encoded_n[0], n[0]))
                #             if len(token_neighbors) >= max_neighbors:
                #                 break
                #     if len(token_neighbors) != 0:
                #         self.neighbor_dict[encoded_token[0]] = token_neighbors
            with open(cache, 'wb') as f:
                pickle.dump(self.neighbor_dict, f)

    def get_neighbors(self, token_id):
        if self.neighbor_dict.__contains__(token_id):
            return self.neighbor_dict[token_id]
        else:
            return []

class RandomReplaceDataset():
    def __init__(self, path, tokenizer, fasttext_path, ratio, th=0.4, max_neighbors=5, data_type='persona'):
        self.fasttext_model = fasttext.load_model(fasttext_path)
        self.data_type = data_type
        self.tokenizer = tokenizer
        self.path = path
        self.max_neighbors = max_neighbors
        self.th = th
        self.ratio = ratio

    def build_dict(self, parsed_data):
        token_set = set()
        for persona in tqdm(parsed_data):
            for k, v in persona.items():
                for sentence in v:
                    if isinstance(sentence, list):
                        sentence = ' '.join(sentence)
                    tokens = sentence.split(' ')
                    for t in tokens:
                        if t not in SPECIAL_TOKENS:
                            if t[-1] in ['.', ',', '!', '?', ';']:
                                token_set.add(t[:-1])
                            elif t.find('.') != -1:
                                ts = t.split('.')
                                for tmp_t in ts:
                                    token_set.add(tmp_t)
                            else:
                                token_set.add(t)
        neighbor_dict = {}
        for token in list(token_set):
            neighbors = self.fasttext_model.get_nearest_neighbors(token)
            neighbor_dict[token] = neighbors
        return neighbor_dict

    def add_tokenized_data(self, chat, dataset):
        persona_info = [self.tokenizer.encode(self.tokenizer.tokenize(s)) for s in chat['persona_info']]
        dialog = []
        for i, replica in enumerate(chat['dialog'], 1):
            dialog.append(self.tokenizer.encode(self.tokenizer.tokenize(replica)))
            if not i % 2:
                if chat['candidates']:
                    candidates_ids = [self.tokenizer.encode(self.tokenizer.tokenize(c)) for c in chat['candidates'][(i - 1) // 2]]
                    dataset.append((persona_info, dialog[:], candidates_ids))
                else:
                    dataset.append((persona_info, dialog[:], []))

    def build_cache(self, cache_path):
        if self.data_type == 'persona':
            parsed_data = FacebookDataset.parse_data(self.path)
        elif self.data_type == 'emoji':
            parsed_data = FacebookDataset.parse_data_emoji(self.path)
        neighbor_dict = self.build_dict(parsed_data)

        dataset = []
        for data in tqdm(parsed_data):
            persona = data['persona_info']
            dialog = data['dialog']
            token_cnt = 0
            persona_len = len(persona)
            token_idx = []
            token_list = []
            for i, p in enumerate(persona):
                tmp_tokens = p[:-1].split(' ')
                tmp_tokens.append('.')
                cur_tokens_list = []
                for j, t in enumerate(tmp_tokens):
                    cur_tokens_list.append(t)
                    token_cnt += 1
                    token_idx.append((i, j))
                token_list.append(cur_tokens_list)
            for i, d in enumerate(dialog):
                if i % 2 != 0:
                    continue
                tmp_tokens = d.split(' ')
                cur_tokens_list = []
                for j, t in enumerate(tmp_tokens):
                    cur_tokens_list.append(t)
                    token_cnt += 1
                    token_idx.append((i//2 + persona_len, j))
                token_list.append(cur_tokens_list)
            replace_num = math.ceil(token_cnt * self.ratio)
            random.shuffle(token_idx)
            idx = 0
            while replace_num > 0:
                replace_idx = token_idx[idx]
                idx += 1
                if idx >= len(token_idx):
                    break
                ori_token = token_list[replace_idx[0]][replace_idx[1]]
                if neighbor_dict.__contains__(ori_token) and neighbor_dict[ori_token][0][0] > self.th:
                    token_list[replace_idx[0]][replace_idx[1]] = neighbor_dict[ori_token][0][1]
                    replace_num -= 1
            self.add_tokenized_data(data, dataset)
            replaced_texts = [' '.join(tokens) for tokens in token_list]
            data['persona_info'] = replaced_texts[:persona_len]
            for i in range(persona_len, len(replaced_texts)):
                data['dialog'][(i - persona_len) * 2] = replaced_texts[i]
            self.add_tokenized_data(data, dataset)
        torch.save(dataset, cache_path)
