import datetime
import logging
import math
import random
import pickle
import spacy

import fasttext
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.utils.data import RandomSampler
from torch.autograd import Variable
from tqdm import tqdm
from transformers.tokenization_gpt2 import GPT2Tokenizer
from transformers.tokenization_openai import OpenAIGPTTokenizer
from transformers.modeling_bert import BertForMaskedLM
from transformers.tokenization_bert import BertTokenizer
from model.seq2seq import TransformerSeq2Seq
from model.seq2seq_vocab import Seq2seqTokenizer
from copy import deepcopy

from .loss import LabelSmoothingLoss
from .optim import Adam
from .optim import NoamOpt
from .utils import pad_sequence
from .utils import repeat_along_dim1

SPECIAL_TOKENS = ['<bos>', '<eos>', '<pad>', '<talker1_bos>', '<talker2_bos>', '<talker1_eos>', '<talker2_eos>',
                  '<info_bos>', '<info_eos>', '.', ',', '?', '!', ':']
MIX_IGNORE_TOKENS = ['.', ',', '!', '?', ';', ':', '-', '*', '=', ')', '(', '\'', '"', ]

class Trainer:
    def __init__(self, model, train_dataset, trainer_config, writer, logger=None, test_dataset=None, valid_dataset=None,
                 n_jobs=0, label_smoothing=0, device=torch.device('cuda'), ignore_idxs=[], local_rank=-1,
                 apex_level=None, apex_loss_scale=None, evaluate_full_sequences=False, full_input=False,
                 max_length=511, max_y_length=80, uncertainty_loss=False, new_dataset=False, best_model_path='',
                 extra_module_lr_rate=1, no_persona=False, mixup=False, mixup_mode='alternate', mixup_dataset=None,
                 mixup_ratio=0.15, bert_mixup=False, replace=False, alpha_lr=1e-4,
                 alpha_weight_decay=1e-3, alpha_clip_grad=2, pointer_gen=False):
        n_gpu = torch.cuda.device_count()
        if logger is None:
            self.logger = logging.getLogger(__file__)
        else:
            self.logger = logger
        self.logger.info("device: {}, distributed training: {}, apex_level: {}, apex_scale_loss: {},  n_gpu: {}".format(
            device, bool(local_rank != -1), apex_level, apex_loss_scale, n_gpu))

        self.train_batch_size = trainer_config.train_batch_size
        self.test_batch_size = trainer_config.test_batch_size
        self.lr = trainer_config.lr
        self.lr_warmup = trainer_config.lr_warmup
        self.weight_decay = trainer_config.weight_decay
        self.batch_split = trainer_config.batch_split
        self.s2s_weight = trainer_config.s2s_weight
        self.lm_weight = trainer_config.lm_weight
        self.risk_weight = trainer_config.risk_weight
        self.hits_weight = trainer_config.hits_weight
        self.single_input = trainer_config.single_input
        self.clip_grad = trainer_config.clip_grad
        self.n_epochs = trainer_config.n_epochs
        self.linear_schedule = trainer_config.linear_schedule
        self.patience = trainer_config.patience
        self.model_saving_interval = trainer_config.model_saving_interval
        self.device = device
        self.ignore_idxs = ignore_idxs
        self.apex_level = apex_level
        self.no_persona = no_persona
        self.evaluate_full_sequences = evaluate_full_sequences
        self.global_step = 0
        self.local_rank = local_rank
        self.full_input = full_input
        self.max_length = max_length
        self.max_y_length = max_y_length
        self.new_dataset = new_dataset
        self.best_ppl = 1e5
        self.best_model_path = best_model_path
        if train_dataset is not None:
            self.negative_samples = train_dataset.negative_samples
        self.mixup_mode = mixup_mode
        self.replace = replace
        self.mixup = mixup
        self.mixup_dataset = mixup_dataset
        self.mixup_ratio = mixup_ratio
        self.model_type = 'pretrain'
        self.patience_cnt = 0
        self.stop_training = False
        self.pointer_gen = pointer_gen

        self.model = model.to(device)
        self.uncertainty_loss = uncertainty_loss

        self.lm_criterion = nn.CrossEntropyLoss(ignore_index=self.model.padding_idx).to(device)
        self.hits_criterion = nn.CrossEntropyLoss().to(device)
        self.criterion = LabelSmoothingLoss(n_labels=self.model.n_embeddings, smoothing=label_smoothing, ignore_index=self.model.padding_idx).to(device)

        param_optimizer = list(self.model.named_parameters())
        # Here we should remove parameters which are not used during to avoid breaking apex with None grads
        self.loss_weight = None
        if self.uncertainty_loss:
            if self.hits_weight > 0:
                loss_weight = torch.zeros(3, device=device)
            else:
                loss_weight = torch.zeros(2, device=device)
            self.loss_weight = ('loss.weight', nn.Parameter(loss_weight))
            param_optimizer.append(self.loss_weight)
        no_decay = ['bias', 'loss']
        if extra_module_lr_rate == 1:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
        else:
            params, extra_params, no_decay_params, extra_no_decay_params = [], [], [], []
            for n, p in param_optimizer:
                if not any(nd in n for nd in no_decay):
                    if 'attention_module' in n:
                        extra_params.append(p)
                    else:
                        params.append(p)
                else:
                    if 'attention_module' in n:
                        extra_no_decay_params.append(p)
                    else:
                        no_decay_params.append(p)

            optimizer_grouped_parameters = [
                {'params': params, 'weight_decay': self.weight_decay},
                {'params': extra_params, 'weight_decay': self.weight_decay, 'extra': True},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ]
            if len(extra_no_decay_params) != 0:
                optimizer_grouped_parameters.append({'params': extra_no_decay_params, 'weight_decay': 0, 'extra': True})

        base_optimizer = Adam(optimizer_grouped_parameters, lr=self.lr)
        assert local_rank == -1 or apex_level is None, 'Distributed model with apex optimization is not supported right now.'
        # self.model, base_optimizer = apex_model(self.model, optimizer=base_optimizer,
        #                                         apex_level=apex_level, apex_loss_scale=apex_loss_scale)

        if not self.linear_schedule:
            self.optimizer = NoamOpt(self.model.embeddings_size, self.lr_warmup, base_optimizer, lr=self.lr,
                                     linear_schedule=False, apex_level=apex_level, loss_weight=self.loss_weight,
                                     extra_module_lr_rate=extra_module_lr_rate)
        else:
            total_steps = len(train_dataset) * self.n_epochs // self.train_batch_size
            if local_rank != -1:
                total_steps = total_steps // torch.distributed.get_world_size()
            self.optimizer = NoamOpt(self.model.embeddings_size, self.lr_warmup, base_optimizer, linear_schedule=True,
                                     lr=self.lr, total_steps=total_steps, apex_level=apex_level, loss_weight=self.loss_weight,
                                     extra_module_lr_rate=extra_module_lr_rate)

        if local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            train_sampler = DistributedSampler(train_dataset)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.train_batch_size // self.batch_split,
                                           sampler=train_sampler,
                                           num_workers=n_jobs, collate_fn=self.collate_func)
        self.train_dataset = train_dataset  # used to sample negative examples
        if test_dataset is not None and local_rank in [-1, 0]:  # only do evaluation on main process
            self.test_dataloader = DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=False,
                                              num_workers=n_jobs, collate_fn=self.collate_func)
        if valid_dataset is not None and local_rank in [-1, 0]:
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.test_batch_size, shuffle=False,
                                              num_workers=n_jobs, collate_fn=self.collate_func)
        self.bert_mixup = bert_mixup
        if bert_mixup:
            self.bert_model = BertForMaskedLM.from_pretrained('./bert_model').to(device)
            self.bert_tokenizer = BertTokenizer.from_pretrained('./bert_model')

        self.vocab = train_dataset.vocab
        self.writer = writer

        if isinstance(self.model, TransformerSeq2Seq):
            self.model_type = 'seq2seq'

    def state_dict(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'global_step': self.global_step}

    def load_state_dict(self, state_dict):
        if state_dict.__contains__('model') and state_dict.__contains__('optimizer'):
            self.model.load_state_dict(state_dict['model'], strict=False)
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.global_step = state_dict['global_step']
        else:
            self.model.load_state_dict(state_dict, strict=False)

    def collate_func(self, data):
        persona_info, h, y, distractors_batch = zip(*data)

        contexts = []

        if max(map(len, persona_info)) > 0:
            persona_info = [torch.tensor(d, dtype=torch.long) for d in persona_info]
            contexts.append(persona_info)

        if max(map(len, h)) > 0:
            h = [torch.tensor(d, dtype=torch.long) for d in h]
            contexts.append(h)

        y_out = [torch.tensor(d, dtype=torch.long) for d in y]

        distractors = [torch.tensor(d, dtype=torch.long) for distractors in distractors_batch for d in distractors]

        if self.single_input:
            # we concatenate all the contexts in y (idem for distractors)
            if self.no_persona:
                for c in contexts[1]:
                    c[0][0] = self.vocab.bos_id
                y_out = [torch.cat(pieces, dim=0) for pieces in zip(*([contexts[1]] + [y_out]))]
                distractors_contexts = []
                for c in contexts[1]:
                    distractors_contexts.extend([c] * self.negative_samples)
                distractors = [torch.cat(pieces, dim=0) for pieces in zip(*([distractors_contexts] + [distractors]))]
                lengths = [(contexts[1][i].size(0), y_out[i].size(0)) for i in range(len(y_out))]
                lengths.extend([(distractors_contexts[i].size(0), distractors[i].size(0)) for i in range(len(distractors))])
                contexts = lengths
            else:
                distractors_contexts = [[], []]
                for i in range(len(contexts[0])):
                    distractors_contexts[0].extend([contexts[0][i]] * self.negative_samples)
                    distractors_contexts[1].extend([contexts[1][i]] * self.negative_samples)
                if self.model_type == 'seq2seq':
                    y_out1 = [torch.cat(pieces, dim=0) for pieces in zip(*(contexts))]
                    distractors1 = [torch.cat(pieces, dim=0) for pieces in zip(*(distractors_contexts))]
                    lengths = [(contexts[0][i].size(0) + contexts[1][i].size(0), y_out[i].size(0)) for i in range(len(y_out))]
                    if len(distractors1) > 0:
                        lengths.extend(
                            [(distractors_contexts[0][i].size(0) + distractors_contexts[1][i].size(0),
                              distractors[i].size(0))
                             for i in range(len(distractors1))])
                    y_out = (y_out1, y_out)
                    distractors = (distractors1, distractors)
                else:
                    y_out = [torch.cat(pieces, dim=0) for pieces in zip(*(contexts + [y_out]))]
                    distractors = [torch.cat(pieces, dim=0) for pieces in zip(*(distractors_contexts + [distractors]))]
                    lengths = [(contexts[0][i].size(0) + contexts[1][i].size(0), y_out[i].size(0)) for i in range(len(y_out))]
                    if len(distractors) > 0:
                        lengths.extend(
                            [(distractors_contexts[0][i].size(0) + distractors_contexts[1][i].size(0),
                              distractors[i].size(0))
                                    for i in range(len(distractors))])
                contexts = lengths
        else:
            if self.full_input:
                y_out = [torch.cat(pieces, dim=0) for pieces in zip(*(contexts + [y_out]))]
                extended_contexts = [[t for t in c for _ in range(len(distractors) // len(y))] for c in contexts]
                distractors = [torch.cat(pieces, dim=0) for pieces in zip(*(extended_contexts + [distractors]))]
                for i, seq in enumerate(y_out):
                    if seq.shape[0] > self.max_length:
                        history_start, history_end = -1, -1
                        for j in range(seq.shape[0]):
                            if history_start == -1 and \
                                    (seq[j][1] == self.vocab.talker1_dialog_id or seq[j][1] == self.vocab.talker2_dialog_id):
                                history_start = j
                            if history_end == -1 and seq[j][1] == self.vocab.sent_dialog_id:
                                history_end = j
                                break
                        history_length = self.max_length - history_start - (seq.shape[0] - history_end)
                        y_out[i] = torch.cat([y_out[i][:history_start], y_out[i][history_end - history_length:]], dim=0)

            # Pad now so we pad correctly when we have only a single input (context concatenated with y)
        if self.single_input:
            if isinstance(y_out, tuple):
                y_out = ([y[-(self.max_length - 1):] for y in y_out[0]], [y[:(self.max_y_length - 1)] for y in y_out[1]])
                distractors = ([d[-(self.max_length - 1):] for d in distractors[0]],
                               [d[:(self.max_length - 1)] for d in distractors[1]])
            else:
                y_out = [y[-(self.max_length - 1):] for y in y_out]
                distractors = [d[-(self.max_length - 1):] for d in distractors]
            contexts = [c if c[1] <= self.max_length - 1 else (c[0] - (c[1] - self.max_length + 1), self.max_length - 1) for c in contexts]
        else:
            y_out = [y[: self.max_length] for y in y_out]
            distractors = [d[: self.max_length] for d in distractors]
            contexts = [c[:self.max_length] for c in contexts]
        # with open('error1.pickle', 'wb') as f:
        #     pickle.dump({'y_out': y_out, 'distractors': distractors}, f)
        if isinstance(y_out, tuple):
            y_out = (pad_sequence(y_out[0], batch_first=True, padding_value=self.model.padding_idx),
                     pad_sequence(y_out[1], batch_first=True, padding_value=self.model.padding_idx))
            distractors = (pad_sequence(distractors[0], batch_first=True, padding_value=self.model.padding_idx),
                           pad_sequence(distractors[1], batch_first=True, padding_value=self.model.padding_idx))
        else:
            y_out = pad_sequence(y_out, batch_first=True, padding_value=self.model.padding_idx)
            distractors = pad_sequence(distractors, batch_first=True, padding_value=self.model.padding_idx)
        if not self.single_input:
            contexts = [pad_sequence(c, batch_first=True, padding_value=self.model.padding_idx) for c in contexts]

        return contexts, y_out, distractors

    def _lm_loss(self, contexts, enc_contexts):
        batch_lm_loss = torch.tensor(0, dtype=torch.float, device=self.device)

        if self.single_input:
            return batch_lm_loss

        for context in contexts:
            enc_context = self.model.encode(context.clone())
            enc_contexts.append(enc_context)

            if self.lm_weight > 0:
                context_outputs = self.model.generate(enc_context[0])
                ignore_mask = torch.stack([context == idx for idx in self.ignore_idxs], dim=-1).any(dim=-1)
                context.masked_fill_(ignore_mask, self.model.padding_idx)
                prevs = context_outputs[:, :-1, :].contiguous()
                nexts = context[:, 1:].contiguous() if context.dim() == 2 else context[:, 1:, 0].contiguous()
                batch_lm_loss += self.lm_criterion(prevs.view(-1, prevs.shape[-1]).float(), nexts.view(-1)) / len(contexts)
        return batch_lm_loss

    def _loss_single(self, targets, distractors, lengths, target_mix_replace, distractor_mix_replace):
        input_ids = targets[:, :, 0].contiguous()
        token_type_ids = targets[:, :, 1].contiguous()
        lm_labels = -100 * torch.ones_like(input_ids)
        mc_token_ids = torch.tensor([l[1] - 1 for l in lengths], device=self.device)
        cur_batch_size = input_ids.size(0)
        for i in range(cur_batch_size):
            lm_labels[i, lengths[i][0] + 1: lengths[i][1]] = targets[i, lengths[i][0] + 1: lengths[i][1], 0].contiguous()
        lm_loss, lm_logits, mc_logits, _ = self.model(input_ids, token_type_ids=token_type_ids, lm_labels=lm_labels,
                mc_token_ids=mc_token_ids[: cur_batch_size], mix_replace=target_mix_replace)
        all_mc_logits = [mc_logits.unsqueeze(-1)]
        if distractors.size()[0] > 0:
            for i in range(self.negative_samples):
                distractor_ids = distractors[cur_batch_size * i: cur_batch_size * (i + 1), :, 0]. contiguous()
                distractor_type_ids = distractors[cur_batch_size * i: cur_batch_size * (i + 1), :, 1]. contiguous()
                distractor_mix_replace_batch = None
                if distractor_mix_replace is not None:
                    distractor_mix_replace_batch = distractor_mix_replace[cur_batch_size * i: cur_batch_size * (i + 1)]
                _, mc_logits, _ = self.model(
                    distractor_ids,
                    token_type_ids=distractor_type_ids,
                    mc_token_ids=mc_token_ids[cur_batch_size * (i + 1): cur_batch_size * (i + 2)],
                    mix_replace=distractor_mix_replace_batch)
                all_mc_logits.append(mc_logits.unsqueeze(-1))
        mc_labels = torch.zeros_like(mc_logits, dtype=torch.long)
        mc_logits = torch.cat(all_mc_logits, dim=-1)
        if self.model.training:
            loss_fct = CrossEntropyLoss()
            mc_loss = loss_fct(mc_logits, mc_labels)
        else:
            mc_loss = torch.sum(torch.max(mc_logits, dim=1)[1] == mc_labels).float() / mc_labels.shape[0]
        return lm_loss, mc_loss

    def _s2s_loss(self, targets, enc_contexts, negative_samples):
        hidden_state, padding_mask = None, None

        nexts = targets[:, 1:].contiguous() if targets.dim() == 2 else targets[:, 1:, 0].contiguous()
        if self.hits_weight > 0 and negative_samples > 0:
            # Keep the hidden states for hits@1 loss
            outputs, hidden_state, padding_mask = self.model.decode(targets[:, :-1].contiguous(), enc_contexts)
        else:
            outputs, _, _ = self.model.decode(targets[:, :-1].contiguous(), enc_contexts)
        if self.full_input:
            for i in range(targets.shape[0]):
                for j in range(targets.shape[1]):
                    if targets[i][j][1] == self.vocab.sent_dialog_id:
                        nexts[i][: j] = self.model.padding_idx
                        break

        outputs = outputs.view(-1, outputs.shape[-1]).float()
        nexts = nexts.view(-1)

        loss = self.criterion(F.log_softmax(outputs, dim=-1), nexts) if self.model.training \
               else self.lm_criterion(outputs, nexts)
        return loss, hidden_state, padding_mask

    def _hist(self, distractors, hidden_state, padding_mask, enc_contexts, negative_samples):
        batch_hits_loss = torch.tensor(0, dtype=torch.float, device=self.device)

        if self.hits_weight == 0 or negative_samples == 0:
            return batch_hits_loss

        extended_contexts = repeat_along_dim1(enc_contexts, negative_samples)
        neg_logits = self.model.decode_classify(distractors, extended_contexts)
        true_logits = self.model.classify(hidden_state, padding_mask)
        clf_logits = torch.cat((true_logits.view(-1, 1), neg_logits.view(-1, negative_samples)), dim=1)
        clf_labels = torch.tensor([0] * len(true_logits), dtype=torch.long, device=self.device)

        batch_hits_loss = self.hits_criterion(clf_logits.float(), clf_labels) if self.model.training else \
                          torch.sum(torch.max(clf_logits, dim=1)[1] == clf_labels).float() / clf_labels.shape[0]

        return batch_hits_loss

    def _risk_loss(self, contexts, targets, enc_contexts, risk_func):

        if risk_func is None or self.risk_weight == 0:
            return torch.tensor(0, dtype=torch.float, device=self.device)

        self.model.eval()  # desactivate dropout

        if self.single_input:
            beam_starts = pad_sequence(contexts, batch_first=True, padding_value=self.model.padding_idx, left=True)
            beams, beam_lens = self.model.beam_search(beam_starts=beam_starts, return_beams=True)
        else:
            beams, beam_lens = self.model.beam_search(enc_contexts=enc_contexts, return_beams=True)

        self.model.train()  # re-activate dropout

        labels = targets if targets.dim() == 2 else targets[:, :, 0]
        labels_lens = labels.ne(self.model.padding_idx).sum(dim=-1)
        labels_start = [context.shape[0] + 1 for context in contexts] if self.single_input else [1] * len(labels)
        labels = [t[s:l - 1].tolist() for t, s, l in zip(labels, labels_start, labels_lens)]

        batch_risks = []
        for b in range(self.model.beam_size):
            predictions = [t[b][1:l[b] - 1].tolist() for t, l in zip(beams, beam_lens)]
            risks = torch.tensor(risk_func(predictions, labels), dtype=torch.float, device=self.device)
            batch_risks.append(risks)
        batch_risks = torch.stack(batch_risks, dim=-1)

        if self.model.dialog_embeddings:
            beams = torch.stack((beams, torch.full_like(beams, self.model.sent_dialog_id)), dim=beams.dim())

        if self.single_input:
            start = beam_starts.size(1)
            beam_starts.unsqueeze_(1)
            beam_starts = beam_starts.repeat([1, self.model.beam_size] + [1] * len(beam_starts.size()[2:])) # tail_dims for dialog_embeddings
            beams = torch.cat((beam_starts, beams), dim=2)

        batch_probas = []
        for b in range(self.model.beam_size):
            inputs = beams[:, b, :-1]
            outputs = beams[:, b, 1:]

            outputs = outputs[:, :, 0] if outputs.dim() == 3 else outputs
            logits = self.model.decode(inputs, enc_contexts)

            probas = F.log_softmax(logits.float(), dim=-1)
            probas = torch.gather(probas, -1, outputs.unsqueeze(-1)).squeeze(-1)
            probas.masked_fill_(outputs.eq(self.model.padding_idx), 0)
            probas = probas[:, start:] if self.single_input else probas

            probas = probas.sum(dim=-1) / beam_lens[:, b].float()

            batch_probas.append(probas)
        batch_probas = torch.stack(batch_probas, dim=-1)
        batch_probas = F.softmax(batch_probas, dim=-1)

        batch_risk_loss = torch.mean((batch_risks * batch_probas).sum(dim=-1))

        return batch_risk_loss

    def random_replace(self, ids, ratio, lengths, th=0.4, max_mix=5):
        def padding_ids(ori_ids, max_len):
            new_ids = [idx for idx in ori_ids]
            cur_idx = 0
            while len(new_ids) < max_len:
                new_ids.append(ori_ids[cur_idx])
                cur_idx += 1
                if cur_idx == len(ori_ids):
                    cur_idx = 0
            return new_ids
        res = []
        for i in range(ids.size(0)):
            if len(ids.size()) > 2:
                cur_ids_list = ids[i][:, 0].tolist()
            else:
                cur_ids_list = ids[i].tolist()
            decoded_tokens = [self.vocab.decode([x], skip_special_tokens=True) for x in cur_ids_list[:lengths[i]]]
            token_new_start = True
            candidate_encoded_ids = []
            cur_token_ids = (1, [])
            for j, token in enumerate(decoded_tokens):
                if isinstance(self.vocab, Seq2seqTokenizer):
                    if len(token) > 0 and token not in MIX_IGNORE_TOKENS:
                        candidate_encoded_ids.append((j, tuple([cur_ids_list[j]])))
                else:
                    if len(token) > 0:
                        if token in MIX_IGNORE_TOKENS:
                            token_new_start = True
                            continue
                        if token[0] == ' ':
                            token_new_start = True
                        if token_new_start:
                            if len(cur_token_ids[1]) > 0:
                                candidate_encoded_ids.append((cur_token_ids[0], tuple(cur_token_ids[1])))
                            cur_token_ids = (j, [])
                            cur_token_ids[1].append(cur_ids_list[j])
                            token_new_start = False
                        else:
                            cur_token_ids[1].append(cur_ids_list[j])
                    else:
                        token_new_start = True
            if len(cur_token_ids[1]) > 0:
                candidate_encoded_ids.append((cur_token_ids[0], tuple(cur_token_ids[1])))
            candidate_idxs = list(range(len(candidate_encoded_ids)))
            random.shuffle(candidate_idxs)
            mix_token_num = max(math.floor(float(len(candidate_encoded_ids)) * ratio), 1)
            if self.bert_mixup:
                cur_res = self.get_bert_replace(decoded_tokens, candidate_idxs, candidate_encoded_ids, mix_token_num)
            else:
                cur_res = self.get_fasttext_repalce(candidate_idxs, candidate_encoded_ids, mix_token_num, padding_ids)
            if self.replace:
                for candidate in cur_res:
                    ids[i][candidate[2]] = candidate[0][0]
            else:
                res.append(cur_res)
        if self.replace:
            return None, ids
        return res, ids

    def get_fasttext_repalce(self, candidate_idxs, candidate_encoded_ids, mix_token_num, padding_ids):
        cur_res = []
        for candidate_idx in candidate_idxs:
            if mix_token_num == 0:
                break
            neighbors = self.mixup_dataset.get_neighbors(candidate_encoded_ids[candidate_idx][1])
            if len(neighbors) > 0:
                start_pos = candidate_encoded_ids[candidate_idx][0]
                max_neighbor_len = max([len(n[0]) for n in neighbors])
                neighbors_ids = [n[0] if len(n[0]) == max_neighbor_len else padding_ids(n[0], max_neighbor_len)
                                 for n in neighbors]
                cur_res.append((
                    torch.tensor(neighbors_ids, dtype=torch.long),
                    torch.tensor([n[1] for n in neighbors], dtype=torch.float),
                    torch.tensor(list(range(start_pos, start_pos + len(candidate_encoded_ids[candidate_idx][1]))),
                                 dtype=torch.long)
                ))
                mix_token_num -= 1
        return cur_res

    def get_bert_replace(self, decoded_tokens, candidate_idxs, candidate_encoded_ids, mix_token_num, max_replace=5):
        input_ids = self.bert_tokenizer.convert_tokens_to_ids(decoded_tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        logits = self.bert_model(input_ids.unsqueeze(0))[0].squeeze(0)
        topk_logits, topk_token_ids = torch.topk(F.softmax(logits, dim=1), k=10)
        cur_res = []
        for candidate_idx in candidate_idxs:
            if mix_token_num == 0:
                break
            cur_logits = topk_logits[candidate_encoded_ids[candidate_idx][0]].tolist()
            cur_tokens = self.bert_tokenizer.convert_ids_to_tokens(
                topk_token_ids[candidate_encoded_ids[candidate_idx][0]].tolist())
            cur_encoded_ids = [self.vocab.encode([token]) for token in cur_tokens]
            cnt = 0
            tmp_replace_ids, tmp_prob = [], []
            for i in range(len(cur_logits)):
                if len(cur_encoded_ids[i]) != 0 and cur_encoded_ids[i][0] != candidate_encoded_ids[candidate_idx][1][0]:
                    tmp_replace_ids.append(cur_encoded_ids[i])
                    tmp_prob.append(cur_logits[i])
                    cnt += 1
                    if cnt >= max_replace:
                        break
            if len(tmp_replace_ids) > 0:
                prob_tensor = torch.tensor(tmp_prob, dtype=torch.float)
                cur_res.append((
                    torch.tensor(tmp_replace_ids, dtype=torch.long),
                    prob_tensor / torch.sum(prob_tensor),
                    torch.tensor([candidate_encoded_ids[candidate_idx][0]], dtype=torch.long)
                ))
                mix_token_num -= 1
        return cur_res

    def optimizer_step(self, lm_loss, risk_loss, hits_loss, s2s_loss, full_loss):
        if self.clip_grad is not None:
            for group in self.optimizer.param_groups:
                nn.utils.clip_grad_norm_(group['params'], self.clip_grad)

        self.optimizer.step()
        self.optimizer.zero_grad()

        global_step = max(self.global_step, 0)
        self.writer.add_scalar("training/lm_loss", lm_loss, global_step=global_step)
        self.writer.add_scalar("training/risk_loss", risk_loss, global_step=global_step)
        self.writer.add_scalar("training/hits_loss", hits_loss, global_step=global_step)
        self.writer.add_scalar("training/s2s_loss", s2s_loss, global_step=global_step)
        self.writer.add_scalar("training/full_loss", full_loss, global_step=global_step)
        self.writer.add_scalar("training/lr", self.optimizer.get_lr(), global_step=global_step)

        self.global_step += 1

    def _eval_train(self, epoch, risk_func=None): # add ppl and hits@1 evaluations
        self.model.train()

        tqdm_data = tqdm(self.train_dataloader, desc='Train (epoch #{})'.format(epoch))
        s2s_loss = 0
        lm_loss = 0
        risk_loss = 0
        hits_loss = 0
        mixup_train = False
        if self.mixup and self.mixup_mode == 'all':
            mixup_train = True
        for i, (contexts, targets, distractors) in enumerate(tqdm_data):
            negative_samples = self.negative_samples
            if not self.single_input:
                contexts, targets, distractors = [c.to(self.device) for c in contexts], targets.to(self.device), \
                                                 distractors.to(self.device)
                enc_contexts = []

                if isinstance(self.model, TransformerSeq2Seq):
                    loss = self.model(contexts, targets)
                    full_loss = (loss / self.batch_split,)
                    s2s_loss = (i * s2s_loss + loss.item()) / (i + 1)
                    tqdm_data.set_postfix({'s2s_loss': s2s_loss})
                else:
                    # lm loss on contexts
                    batch_lm_loss = self._lm_loss(contexts, enc_contexts)
                    # batch_lm_loss = (self._lm_loss(enc_persona_generated, persona.clone()) + self._lm_loss(enc_dialog_generated, dialog.clone())) / 2

                    # s2s loss on targets
                    batch_s2s_loss, hidden_state, padding_mask = self._s2s_loss(targets, enc_contexts, negative_samples)

                    # hits@1 loss on distractors and targets
                    batch_hits_loss = self._hist(distractors, hidden_state, padding_mask, enc_contexts, negative_samples)

                    # risk loss
                    batch_risk_loss = self._risk_loss(contexts, targets, enc_contexts, risk_func)
                    full_loss = (self.lm_weight * batch_lm_loss / self.batch_split,
                                 self.risk_weight * batch_risk_loss / self.batch_split,
                                 self.hits_weight * batch_hits_loss / self.batch_split,
                                 self.s2s_weight * batch_s2s_loss / self.batch_split)
                    lm_loss = (i * lm_loss + batch_lm_loss.item()) / (i + 1)
                    s2s_loss = (i * s2s_loss + batch_s2s_loss.item()) / (i + 1)
                    risk_loss = (i * risk_loss + batch_risk_loss.item()) / (i + 1)
                    hits_loss = (i * hits_loss + batch_hits_loss.item()) / (i + 1)
                    tqdm_data.set_postfix({'lm_loss': lm_loss, 's2s_loss': s2s_loss,
                                           'risk_loss': risk_loss, 'hits_loss': hits_loss})
            else:
                if isinstance(self.model, TransformerSeq2Seq):
                    input_ids, labels, lengths = targets[0].to(self.device), targets[1].to(self.device), contexts
                    input_ids_replace, labels_replace = None, None,
                    # if i == 4:
                    #     labels = torch.tensor([[68, 19196]])
                    #     input_ids = input_ids[6:7]
                    # print(str(input_ids.size(1)) + ' ' + str(labels.size(1)))
                    if mixup_train:
                        input_ids_replace, input_ids = self.random_replace(input_ids, self.mixup_ratio, [l[0] for l in lengths])
                        labels_replace, labels = self.random_replace(labels, self.mixup_ratio, [l[1] for l in lengths])
                        if not self.replace:
                            input_ids_replace = [
                                [(y[0].to(self.device), y[1].to(self.device), y[2].to(self.device)) for y in x]
                                for x in input_ids_replace]
                            labels_replace = [
                                [(y[0].to(self.device), y[1].to(self.device), y[2].to(self.device)) for y in x]
                                for x in labels_replace]
                    loss = self.model(input_ids, labels, input_ids_replace, labels_replace)
                    if isinstance(loss, tuple):
                        full_loss = (loss[0] / self.batch_split, )
                        s2s_loss = (i * s2s_loss + loss[1].item()) / (i + 1)
                    else:
                        full_loss = (loss / self.batch_split, )
                        s2s_loss = (i * s2s_loss + loss.item()) / (i + 1)
                    tqdm_data.set_postfix({'s2s_loss': s2s_loss})
                else:
                    # with open('error2.pickle', 'wb') as f:
                    #     pickle.dump({'targets': targets, 'distractors': distractors, 'lengths': contexts}, f)
                    targets_mix_replace, distractors_mix_replace = None, None
                    # start_time = datetime.datetime.now()
                    if mixup_train:
                        targets_mix_replace, targets = self.random_replace(targets, self.mixup_ratio,
                                                                           [t.size(0) for t in targets])
                        distractors_mix_replace, distractors = self.random_replace(distractors, self.mixup_ratio,
                                                                                   [d.size(0) for d in distractors])
                        if not self.replace:
                            targets_mix_replace = [
                                [(y[0].to(self.device), y[1].to(self.device), y[2].to(self.device)) for y in x]
                                for x in targets_mix_replace]
                            distractors_mix_replace = [
                                [(y[0].to(self.device), y[1].to(self.device), y[2].to(self.device)) for y in x]
                                for x in distractors_mix_replace]
                        # targets_mix_replace = [[(y[0].to(self.device), y[1].to(self.device), y[2].to(self.device)) for y in x]
                        #                        for x in targets_mix_replace]
                        # distractors_mix_replace = [[(y[0].to(self.device), y[1].to(self.device), y[2].to(self.device)) for y in x]
                        #                        for x in distractors_mix_replace]
                    # end_time = datetime.datetime.now()
                    # during = end_time - start_time
                    # print('mix up time: ' + str(during.microseconds))
                    targets, distractors, lengths = targets.to(self.device), distractors.to(self.device), contexts

                    batch_s2s_loss, batch_hits_loss = self._loss_single(targets, distractors, lengths,
                                                                        targets_mix_replace, distractors_mix_replace)
                    full_loss = (self.s2s_weight * batch_s2s_loss / self.batch_split,
                                 self.hits_weight * batch_hits_loss / self.batch_split)
                    s2s_loss = (i * s2s_loss + batch_s2s_loss.item()) / (i + 1)
                    hits_loss = (i * hits_loss + batch_hits_loss.item()) / (i + 1)
                    tqdm_data.set_postfix({'s2s_loss': s2s_loss, 'hits_loss': hits_loss})

            # optimization
            full_loss = tuple(filter(lambda x: x.requires_grad, full_loss))
            full_loss = self.optimizer.backward(full_loss)
            if self.pointer_gen and (torch.isnan(self.model.generator.p_gen_linear._parameters['weight']._grad[0][0]) or \
                torch.isinf(self.model.generator.p_gen_linear._parameters['weight']._grad[0][0])):
                self.optimizer.zero_grad()
                self.logger.info('Abnormal gradient')
            # print(self.model.generator.p_gen_linear._parameters['weight']._grad[0][0])
            if self.mixup and self.mixup_mode == 'random':
                if torch.randint(0, 10, (1,)).item() % 2 == 0:
                    mixup_train = True
                else:
                    mixup_train = False

            # print(self.model.generator.p_gen_linear._parameters['weight']._grad)
            if (i + 1) % self.batch_split == 0:
                self.optimizer_step(lm_loss, risk_loss, hits_loss, s2s_loss, full_loss)
                if self.mixup and self.mixup_mode == 'alternate':
                    mixup_train = not mixup_train
        if (i + 1) % self.batch_split != 0:
            self.optimizer_step(lm_loss, risk_loss, hits_loss, s2s_loss, full_loss)

    def _get_eval_loss(self, contexts, targets, distractors, metrics, index):
        lengths, enc_contexts = None, []
        if self.single_input:
            if isinstance(self.model, TransformerSeq2Seq):
                input_ids, labels, lengths = targets[0].to(self.device), targets[1].to(self.device), contexts
                batch_s2s_loss = self.model(input_ids, labels)
                if isinstance(batch_s2s_loss, tuple):
                    batch_s2s_loss = batch_s2s_loss[1]
                batch_hits_acc = torch.tensor(0, dtype=torch.float)
            else:
                targets, distractors, lengths = targets.to(self.device), distractors.to(self.device), contexts
                batch_s2s_loss, batch_hits_acc = self._loss_single(targets, distractors, lengths, None, None)
        else:
            contexts, targets, distractors = [c.to(self.device) for c in contexts], targets.to(self.device), \
                                             distractors.to(self.device)
            if isinstance(self.model, TransformerSeq2Seq):
                batch_s2s_loss, enc_contexts = self.model(contexts, targets, return_encoded=True)
                batch_hits_acc = torch.tensor(0, dtype=torch.float)
            else:

                # lm loss
                batch_lm_loss = self._lm_loss(contexts, enc_contexts)
                metrics['lm_loss'] = (metrics['lm_loss'] * index + batch_lm_loss.item()) / (index + 1)

                # s2s loss on targets
                batch_s2s_loss, hidden_state, padding_mask = self._s2s_loss(targets, enc_contexts,
                                                                            self.negative_samples)
                # hits@1 loss on distractors and targets
                batch_hits_acc = self._hist(distractors, hidden_state, padding_mask,
                                            enc_contexts, self.negative_samples)
                metrics['lm_ppl'] = (metrics['lm_ppl'] * index + math.exp(batch_lm_loss)) / (index + 1)

        metrics['s2s_loss'] = (metrics['s2s_loss'] * index + batch_s2s_loss.item()) / (index + 1)
        metrics['hits_acc'] = (metrics['hits_acc'] * index + batch_hits_acc.item()) / (index + 1)
        metrics['s2s_ppl'] = (metrics['s2s_ppl'] * index + math.exp(batch_s2s_loss)) / (index + 1)
        return metrics, lengths, enc_contexts

    def _get_eval_predictions(self, contexts, targets, lengths, enc_contexts, metrics, metric_funcs,
                             external_metrics_func, index):
        string_references, string_predictions = [], []
        if self.evaluate_full_sequences:
            if self.single_input:
                if isinstance(self.model, TransformerSeq2Seq):
                    labels = targets[1]
                else:
                    labels = []
                    for i in range(targets.shape[0]):
                        labels.append(targets[i, lengths[i][0] + 1: lengths[i][1], 0])
                    labels = pad_sequence(labels, batch_first=True, padding_value=self.model.padding_idx)
            elif not self.full_input:
                labels = targets if targets.dim() == 2 else targets[:, :, 0]
            else:
                labels = []
                new_targets = []
                for i in range(targets.shape[0]):
                    label_start, label_end = -1, targets.shape[1]
                    for j in range(targets.shape[1]):
                        if targets[i][j][1] == self.model.sent_dialog_id and label_start == -1:
                            label_start = j
                        if targets[i][j][1] == self.model.padding_idx:
                            label_end = j
                            break
                    labels.append(targets[i, label_start: label_end, 0])
                    new_targets.append(targets[i, : label_start])
                labels = pad_sequence(labels, batch_first=True, padding_value=self.model.padding_idx)
                targets = pad_sequence(new_targets, batch_first=True, padding_value=self.model.padding_idx,
                                       left=True)
            if self.single_input:
                predictions = []
                if isinstance(self.model, TransformerSeq2Seq):
                    input_ids = targets[0].to(self.device)
                    predictions = self.model.inference(input_ids)
                else:
                    for i in range(targets.size(0)):
                        input_ids = targets[i, :lengths[i][0] + 1, 0].to(self.device)
                        token_type_ids = targets[i, :lengths[i][0] + 1, 1].to(self.device)
                        prediction = self.model.inference(input_ids, token_type_ids)
                        predictions.append(prediction)
            else:
                if isinstance(self.model, TransformerSeq2Seq):
                    predictions = self.model.inference(contexts, encoder_outputs=enc_contexts)
                else:
                    if self.full_input:
                        predictions = self.model.inference(beam_starts=targets, enc_contexts=enc_contexts)
                    else:
                        predictions = self.model.inference(enc_contexts=enc_contexts)

            labels_lens = labels.ne(self.model.padding_idx).sum(dim=-1)
            if not self.single_input:
                labels_start = [1] * len(targets)
                labels = [t[s:l - 1].tolist() for t, s, l in zip(labels, labels_start, labels_lens)]
            else:
                labels = [t[: l - 1].tolist() for t, l in zip(labels, labels_lens)]

            for name, func in metric_funcs.items():
                score = func(predictions, labels)
                metrics[name] = (metrics[name] * index + score) / (index + 1)

            if external_metrics_func:
                # Store text strings for external metrics
                if isinstance(self.vocab, OpenAIGPTTokenizer) or isinstance(self.vocab, GPT2Tokenizer) or \
                        isinstance(self.vocab, Seq2seqTokenizer):
                    string_references = list(
                        self.vocab.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False) for t in
                        labels)
                    string_predictions = list(
                        self.vocab.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False) for t in
                        predictions)
                else:
                    string_references = list(self.vocab.ids2string(t) for t in labels)
                    string_predictions = list(self.vocab.ids2string(t) for t in predictions)
                string_predictions = [x.replace('\n', ' ') for x in string_predictions]
        return string_predictions, string_references

    def _eval_test(self, metric_funcs={}, external_metrics_func=None, epoch=-1, inference=False, is_best=False,
                   raw_entail_data=None):
        with torch.no_grad():
            self.model.eval()
            if epoch == -1:
                tqdm_data = tqdm(self.test_dataloader, desc='Test')
                self.logger.info('Starting testing on Test dataset')
            else:
                tqdm_data = tqdm(self.valid_dataloader, desc='Test')
                self.logger.info('Starting testing on Valid dataset')
            metrics = {name: 0 for name in ('s2s_loss', 'lm_loss', 'hits_acc', 'lm_ppl', 's2s_ppl') + tuple(metric_funcs.keys())}
            full_predictions, full_references = [], []
            for i, (contexts, targets, distractors) in enumerate(tqdm_data):
                '''Get the loss, ppl for each batch'''
                metrics, lengths, enc_contexts = self._get_eval_loss(contexts, targets, distractors, metrics, i)
                # full sequence loss
                cur_predictions, cur_references = self._get_eval_predictions(contexts, targets, lengths, enc_contexts,
                             metrics, metric_funcs, external_metrics_func, i)
                full_predictions.extend(cur_predictions)
                full_references.extend(cur_references)
                tqdm_data.set_postfix(dict(**metrics))
            if raw_entail_data is not None:
                external_metrics_func(full_predictions, raw_entail_data)
                return

            if external_metrics_func and self.evaluate_full_sequences:
                external_metrics = external_metrics_func(full_references, full_predictions, epoch, is_best)
                metrics.update(external_metrics)

            # logging
            global_step = max(self.global_step, 0)
            if self.writer is not None:
                for key, value in metrics.items():
                    self.writer.add_scalar("eval/{}".format(key), value, global_step=global_step)
            self.logger.info(metrics)

            if epoch != -1:
                if metrics['s2s_ppl'] < self.best_ppl:
                    self.logger.info('Current ppl BEATS the previous best one, previous best is %.5f', self.best_ppl)
                    self.best_ppl = metrics['s2s_ppl']
                    torch.save(self.model.state_dict(), self.best_model_path)
                    self.logger.info('Best model is saved on epoch %d', epoch)
                else:
                    self.patience_cnt += 1
                    self.logger.info('Current ppl CANNOT BEATS the previous best one, previous best is %.5f', self.best_ppl)
                    if self.patience > 0 and self.patience_cnt > self.patience:
                        self.stop_training = True
            if epoch % self.model_saving_interval == 0 and epoch >= self.model_saving_interval and \
                    self.model_type in ['seq2seq']:
                torch.save(self.model.state_dict(), self.best_model_path + '_' + str(epoch))

    def _build_split_data_list(self, targets, distractors, lengths, distractor_lengths, split_batch_size):
        split_targets, split_distractors, split_lengths = [], [], []
        batch_size = targets.size(0)
        i = 0
        while i * split_batch_size < batch_size:
            split_targets.append(targets[i * split_batch_size: (i + 1) * split_batch_size])
            split_distractors.append(distractors[split_batch_size * i * self.negative_samples: split_batch_size * (
                        i + 1) * self.negative_samples])
            split_lengths.append(lengths[split_batch_size * i: split_batch_size * (i + 1)] +
                                 distractor_lengths[split_batch_size * i * self.negative_samples: split_batch_size * (
                                             i + 1) * self.negative_samples])
            i += 1
        return split_targets, split_distractors, split_lengths

    def _concat(self, xs):
        return torch.cat([x.view(-1) for x in xs])

    def get_hessian_vector_product(self, weight_grads, train_targets, train_distractors, train_lengths, r=5):
        R = r / (self._concat(weight_grads).norm() + 1)
        for p, v in zip(self.model.parameters(), weight_grads):
            p.data.add_(R, v)
        grads_p, grads_n = None, None
        for (targets, distractors, lengths) in zip(train_targets, train_distractors, train_lengths):
            targets, distractors = targets.to(self.device), distractors.to(self.device)
            batch_s2s_loss, batch_hits_loss = self._loss_single(targets, distractors, lengths, None, None)
            loss = self.s2s_weight * batch_s2s_loss / self.batch_split + \
                   self.hits_weight * batch_hits_loss / self.batch_split
            if grads_p is None:
                grads_p = list(torch.autograd.grad(loss, self.model.get_alpha_parameters(), allow_unused=True))
                self._clip_grad_norm(grads_p, self.alpha_clip_grad)
            else:
                tmp_g = list(torch.autograd.grad(loss, self.model.get_alpha_parameters(), allow_unused=True))
                self._clip_grad_norm(tmp_g, self.alpha_clip_grad)
                for i, g in enumerate(tmp_g):
                    grads_p[i] += g
        grads_p = tuple(grads_p)

        for p, v in zip(self.model.parameters(), weight_grads):
            p.data.sub_(2 * R, v)
        for (targets, distractors, lengths) in zip(train_targets, train_distractors, train_lengths):
            targets, distractors = targets.to(self.device), distractors.to(self.device)
            batch_s2s_loss, batch_hits_loss = self._loss_single(targets, distractors, lengths, None, None)
            loss = self.s2s_weight * batch_s2s_loss / self.batch_split + \
                   self.hits_weight * batch_hits_loss / self.batch_split
            if grads_n is None:
                grads_n = list(torch.autograd.grad(loss, self.model.get_alpha_parameters(), allow_unused=True))
                self._clip_grad_norm(grads_n, self.alpha_clip_grad)
            else:
                tmp_g = list(torch.autograd.grad(loss, self.model.get_alpha_parameters(), allow_unused=True))
                self._clip_grad_norm(tmp_g, self.alpha_clip_grad)
                for i, g in enumerate(tmp_g):
                    grads_n[i] += g
        grads_n = tuple(grads_n)
        for p, v in zip(self.model.parameters(), weight_grads):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)], grads_p, grads_n, R

    def _clip_grad_norm(self, grads, max_norm, norm_type=2):
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        if norm_type == float('inf'):
            total_norm = max(grad.data.abs().max() for grad in grads)
        else:
            total_norm = 0
            for grad in grads:
                grad_norm = grad.data.norm(norm_type)
                total_norm += grad_norm ** norm_type
            total_norm = total_norm ** (1. / norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for grad in grads:
                grad.data.mul_(clip_coef)
        return total_norm

    def test_attention(self, inference_mode=False):
        with torch.no_grad():
            self.model.eval()
            tqdm_data = tqdm(self.test_dataloader, desc='Test')
            self.logger.info('Starting attention testing on Test dataset')
            all_attention = []
            all_attention_inference = []
            loss = 0
            for i, (contexts, targets, distractors) in enumerate(tqdm_data):
                if isinstance(self.model, TransformerSeq2Seq):
                    input_ids, labels, lengths = targets[0].to(self.device), targets[1].to(self.device), contexts
                    batch_loss, attention = self.model(input_ids, labels, get_attention=True)
                    _, attn_list = self.model.inference(input_ids, get_attention=True)
                    for j in range(len(lengths)):
                        all_attention.append(attention[j, :, :lengths[j][1], :lengths[j][0]].detach().cpu())
                        all_attention_inference.append(attn_list[j][:, :, :lengths[j][0]].detach().cpu())
                else:
                    targets, distractors, lengths = targets.to(self.device), distractors.to(self.device), contexts
                    input_ids = targets[:, :, 0].contiguous()
                    token_type_ids = targets[:, :, 1].contiguous()
                    lm_labels = -100 * torch.ones_like(input_ids)
                    mc_token_ids = torch.tensor([l[1] - 1 for l in lengths], device=self.device)
                    cur_batch_size = input_ids.size(0)
                    for i in range(cur_batch_size):
                        lm_labels[i, lengths[i][0] + 1: lengths[i][1]] = targets[i, lengths[i][0] + 1: lengths[i][1],
                                                                         0].contiguous()
                    batch_loss, lm_logits, mc_logits, _, attn = self.model(input_ids, token_type_ids=token_type_ids,
                                                                  lm_labels=lm_labels,
                                                                  mc_token_ids=mc_token_ids[: cur_batch_size])
                    for j in range(len(lengths)):
                        all_attention.append(attn[j, :, lengths[j][0]: lengths[j][1], :lengths[j][0]].detach().cpu())

                loss = (loss * i + batch_loss) / (i + 1)
                tqdm_data.set_postfix({'loss': loss.item()})
        return loss, all_attention, all_attention_inference

    def test(self, metric_funcs={}, external_metrics_func=None, epoch=-1, inference=False, raw_entail_data=None):
        if hasattr(self, 'valid_dataloader') or hasattr(self, 'test_dataloader'):
            self._eval_test(metric_funcs, external_metrics_func, epoch, inference, raw_entail_data=raw_entail_data)
            if epoch == -1 and not inference:
                self.logger.info('Loading the best model...')
                state_dict = torch.load(self.best_model_path, map_location=self.device)
                if state_dict.__contains__('model'):
                    self.model.load_state_dict(state_dict['model'], strict=False)
                else:
                    self.model.load_state_dict(state_dict)
                self._eval_test(metric_funcs, external_metrics_func, epoch, inference, is_best=True)

    def train(self, after_epoch_funcs=[], risk_func=None):
        for epoch in range(1, self.n_epochs + 1):
            self.logger.info('===============================')
            self.logger.info('Start training on Epoch %d', epoch)
            self._eval_train(epoch, risk_func)
            # self._eval_test()

            for func in after_epoch_funcs:
                func(epoch)
            self.logger.info('End training on Epoch %d', epoch)
            self.logger.info('===============================')
            if self.stop_training:
                self.logger.info('Training will be STOPPED in advance due to exceeding patience number')
                break

        for func in after_epoch_funcs:
            func(-1)
