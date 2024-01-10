#!/usr/bin/env python3

import argparse
import random
import json
import logging
import pandas as pd
from scipy import stats
import spacy
import time
from tqdm import tqdm, trange

import torch
import torch.nn.functional as F
import numpy as np
import os

from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer, OpenAIGPTTokenizer

DATA_DIR = '../data'.format(os.getenv('HOME'))
q_sep = 'Q'
a_sep = 'A'

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


# Copied from examples/retrieve_and_edit.py (7/18)
def clean_up_tokenization_spaces(out_string):
    """Converts an output string (de-BPE-ed) using de-tokenization algorithm from OpenAI GPT."""
    out_string = out_string.replace('<unk>', '')
    out_string = out_string.replace(' .', '.').replace(' ?', '?').replace(' !', '!').replace(' ,', ','
            ).replace(" ' ", "'").replace(" n't", "n't").replace(" 'm", "'m").replace(" do not", " don't"
            ).replace(" 's", "'s").replace(" 've", "'ve").replace(" 're", "'re")
    return out_string


# Copied from examples/run_lm.py (7/21)
def load_dataset(split, task_name, debug=False, seed=42):
    """ Output a list of tuples(story, 1st continuation, 2nd continuation, label) """
    assert split in {'train', 'dev'}, 'Split "{}" not yet supported'.format(split)
    examples = []  # Fill examples based on task_name
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

    def format_text(text):
        """Standardizes text using OpenAI GPT's tokenizer (includes lowercasing)"""
        return tokenizer.decode(tokenizer.encode(text.strip())).strip()

    if task_name == 'rocstories':
        file_version = 'test' if split == 'dev' else 'val'
        dataset_path = '{0}/rocstories/cloze_test_{1}__spring2016 - cloze_test_ALL_{1}.csv'.format(
            DATA_DIR, file_version)
        with open(dataset_path, encoding='utf_8') as f:
            f = csv.reader(f)
            next(f)  # Skip the first line
            for line in tqdm(f):
                examples.append((' '.join(line[1:5]), line[5], line[6], int(line[-1])-1))

    elif task_name == 'sqa.q-subqs':
        sqa_split = 'test' if split == 'dev' else 'train'
        wtq_split = 'pristine-unseen-tables' if split == 'test' else 'training'

        df_subq = pd.read_csv('{}/sqa/{}.tsv'.format(DATA_DIR, sqa_split),
                              delimiter='\t', encoding='utf-8')
        df_q = pd.read_csv('{}/WikiTableQuestions/data/{}.tsv'.format(DATA_DIR, wtq_split),
                           delimiter='\t', encoding='utf-8')

        qids_with_subqs = list(set(df_subq['id']))
        qids_with_subqs.sort()

        for qid in tqdm(qids_with_subqs):
            qid = qid.replace('ns', 'nt')
            assert len(df_q[df_q.id == qid].utterance.values) > 0, 'Invalid QID: {}'.format(qid)
            q = df_q[df_q.id == qid].utterance.values[0]
            df_subq_qid = df_subq[df_subq.id == qid]
            annotators = list(set(df_subq_qid.annotator))
            annotators.sort()
            for annotator in annotators:
                df_subq_qid_annotator = df_subq_qid[df_subq_qid.annotator == annotator]
                positions = list(set(df_subq_qid_annotator.position))
                positions.sort()
                subqs = [df_subq_qid_annotator[df_subq_qid_annotator.position == position].question.values[0].strip()
                         for position in positions]
                example_tokens = [q] + subqs
                if '?' not in example_tokens:
                    print(example_tokens)
                example = ' '.join(example_tokens).strip()
                examples.append(example)

    elif task_name in {'squad.q', 'squad.q-q'}:
        file_path = '{}/squad/{}-v2.0.json'.format(DATA_DIR, split)
        with open(file_path, 'r') as f:
            data = json.load(f)

        shuffler = random.Random(seed)
        for article in tqdm(data['data']):
            for para in article['paragraphs']:
                qs = [format_text(qa['question']) for qa in para['qas']]
                if task_name == 'squad.q':
                    examples += qs
                elif task_name == 'squad.q-q':
                    shuffler.shuffle(qs)
                    if (len(qs) % 2) == 1:
                        # qs.append('')  # Use for generative model? Doesn't encourage repeating the original Q.
                        qs.append(qs[-1])  # Use for ranking model. Pair last Q with itself if it would be unpaired.
                    for q1, q2 in zip(qs[::2], qs[1::2]):
                        examples.append((q1 + ' ' + q2).strip())
                else:
                    raise NotImplementedError(task_name)
            if debug and (len(examples) > 100):
                break

    elif task_name == 'squad.sf-q':
        with open('{}/squad/{}-v2.0.json'.format(DATA_DIR, split), 'r') as f:
            data = json.load(f)

        # Sentence tokenization
        nlp = spacy.load("en_core_web_sm")
        for article in tqdm(data['data']):
            for para in article['paragraphs']:
                nlp_para = nlp(para['context'])
                para_sents = list(nlp_para.sents)
                for qa in para['qas']:
                    if qa['is_impossible']:
                        continue  # Only generate answerable Qs
                    # Find sentence containing answer
                    ans_dict = qa['answers'][0]  # Find supporting fact based on first answer only
                    ans_start = ans_dict['answer_start']
                    ans_end = ans_start + len(ans_dict['text']) - 1  # Inclusive
                    sf_start, sf_end = None, None  # Supporting facts sometimes span multiple sentences
                    for sent in para_sents:
                        if (sent.start_char <= ans_start) and (ans_start < sent.end_char):
                            sf_start = sent.start_char
                        if (sent.start_char <= ans_end) and (ans_end < sent.end_char):
                            sf_end = sent.end_char
                        if (sf_start is not None) and (sf_end is not None):
                            break
                    assert (sf_start is not None) and (sf_end is not None), 'Answer sent not found'
                    answer_sf = para['context'][sf_start: sf_end]
                    assert ans_dict['text'] in answer_sf, \
                        'Answer text "{}" not in answer sentence "{}"'.format(ans_dict['text'], answer_sf)
                    examples.append(format_text(answer_sf) + ' ' + q_sep + ' ' +
                                    format_text(qa['question']) + ' ' + a_sep)

    elif task_name in {'hotpot.q-subqs', 'hotpot.q-subqs.comparison'}:
        if task_name == 'hotpot.q-subqs.comparison':
            decomposition_types = ['comparison']
        elif task_name == 'hotpot.q-subqs':
            decomposition_types = ['intersec', 'bridge', 'comparison']
        else:
            raise NotImplementedError(task_name)

        # Load types of each question
        with open('{}/hotpot-all/{}.json'.format(DATA_DIR, split)) as f:
            full_data = json.load(f)
        id_to_qtype = {}
        for example in full_data['data']:
            qa = example['paragraphs'][0]['qas'][0]
            id_to_qtype[qa['id']] = qa['type']

        # Load sub-Qs
        data = {}
        for decomposition_type in decomposition_types:
            filepath = '{}/decomposition-data-nq-version/decomposition-{}-{}-nq.json'.format(
                DATA_DIR, decomposition_type, split)
            with open(filepath, 'r') as f:
                data.update({qid: qinfo for qid, qinfo in json.load(f).items() if id_to_qtype[qid] == 'comparison'})

        for qid, q in tqdm(data.items()):
            if 'subquestions' in q:
                q['subquestion1'], q['subquestion2'] = q['subquestions']
            example_text = ''
            for qtype in ['question', 'subquestion1', 'subquestion2']:
                example_text += q_sep + ' ' + format_text(q[qtype]) + ' '
            if 'op' in q:
                op = q['op'].lower().replace('_', ' ').capitalize()
                example_text += q_sep + ' ' + format_text(op) + ' '
            examples.append(example_text.replace('[ answer ]', 'ANSWER') + q_sep)

    elif task_name == 'hotpot.q-sfs-a':
        with open('{}/hotpot-orig/hotpot_{}_v1.json'.format(
                DATA_DIR, 'train' if split == 'train' else 'dev_distractor'), 'r') as f:
            hotpot_orig = json.load(f)

        missing_sfs = 0
        for example in tqdm(hotpot_orig):
            example_text = example['question'].strip()
            prev_sf_title = ''
            prev_sf_sent_index = -1
            for sf_title, sf_sent_index in example['supporting_facts']:
                for context_title, context_sents in example['context']:
                    if context_title == sf_title:
                        if sf_sent_index >= len(context_sents):
                            missing_sfs += 1
                            continue
                        if sf_title == prev_sf_title:
                            if sf_sent_index == (prev_sf_sent_index + 1):
                                join_text = ' '
                            else:
                                join_text = ' ... '
                        else:
                            join_text = ' [' + sf_title.strip() + '] '
                        example_text += join_text + context_sents[sf_sent_index].strip()
                prev_sf_title = sf_title
                prev_sf_sent_index = sf_sent_index
            example_text = format_text(example_text) + ' ' + a_sep + ' ' + format_text(example['answer']) + ' ' + q_sep
            examples.append(example_text)
        print('missing_sfs', missing_sfs)
        assert missing_sfs < 100, 'Too many missing_sfs ({})'.format(missing_sfs)

    elif task_name == 'hotpot.subqs-subas-q-a':
        num_shards = 100 if split == 'train' else 10
        data = {'data': []}
        data_subas = {}
        for shard_no in range(num_shards):
            file_prefix = 'comparison_decomposed_{}_generations.num_shards={}.shard_no={}'.format(
                split, num_shards, shard_no)
            with open('{}/decomposed-predictions/{}.json'.format(DATA_DIR, file_prefix), 'r') as f:
                data['data'] += json.load(f)['data']
            with open('../DecompRC/DecompRC/out/hotpot/{}.nbest_predictions.json'.format(file_prefix)) as f:
                data_subas.update(json.load(f))

        print('Loading recomposition QA examples...')
        for example in tqdm(data['data']):
            qid = example['paragraphs'][0]['qas_orig'][0]['id']
            question = example['paragraphs'][0]['qas_orig'][0]['question']
            answer = example['paragraphs'][0]['qas_orig'][0]['final_answers'][0]
            subqs = [qa['question'] for qa in example['paragraphs'][0]['qas']]
            subas = []
            if len(subqs) != 2:
                continue  # TODO: Make this fail gracefully: ~6 bad splits, ~12 repetition in sub-question (in train)
            for i in range(len(subqs)):
                subqid = qid + '-' + str(i)
                if subqid in data_subas:
                    subas.append(data_subas[subqid][0]['text'])
            if len(subqs) == len(subas):
                example_text = ''
                for q, a in zip(subqs + [question], subas + [answer]):
                    example_text += q_sep + ' ' + format_text(q) + ' ' + a_sep + ' ' + format_text(a) + ' '
                examples.append(example_text + q_sep)

    else:
        raise NotImplementedError(task_name)

    print('Read {} examples.'.format(len(examples)))
    assert len(examples) > 0, 'Error: Read 0 examples.'
    return examples


def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)


def sample_sequence(model, length, task_name=None, start_token=None, batch_size=None, context=None, temperature=1,
                    top_k=0, device='cuda', sample=True, end_token=None):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    prev = context
    output = context
    past = None
    with torch.no_grad():
        for i in range(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
            if end_token is not None:  # Decode until final token generated
                if task_name == 'hotpot.q-subqs.comparison':
                    num_req_end_tokens = 4
                elif task_name == 'hotpot.subqs-subas-q-a':
                    num_req_end_tokens = 3
                elif task_name in {'hotpot.q-sfs-a', 'squad.sf-q'}:
                    num_req_end_tokens = 1
                else:
                    raise NotImplementedError(task_name)
                if ((output == end_token).sum(1) >= num_req_end_tokens).all():
                    break
    return output


def run_model():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='gpt2',
                        help='pretrained model name or path to local checkpoint')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--length", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
    parser.add_argument('--context_path', type=str, default=None,
                        help='path to contexts to condition on (SQuAD-formatted json).')
    parser.add_argument("--num_shards", default=1, type=int, required=False,
                        help="# of total data splits for distributed eval")
    parser.add_argument("--shard_no", default=0, type=int, required=False, help="Distributed eval data split index.")
    parser.add_argument("--no_task", action='store_true', help="No decoding task (use interactive).")
    args = parser.parse_args()
    print(args)

    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    args_path = os.path.join(args.model_name_or_path, 'training_args.bin')
    model_args = torch.load(args_path) if os.path.exists(args_path) else None
    model.half()
    model.to(device)
    model.eval()

    if args.length == -1:
        args.length = model.config.n_ctx // 2
    elif args.length > model.config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)

    format_tokenizer = None
    if ('squad' in args.model_name_or_path) or ('hotpot' in args.model_name_or_path):
        format_tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

    task_name = None if ((model_args is None) or args.no_task) else model_args.task_name
    if task_name is not None:
        # Backward-compatibility for task_name
        if task_name == 'hotpotqa-recomposition-supporting-fact':
            task_name = 'hotpot.q-sfs-a'
        elif task_name == 'hotpotqa-recomposition':
            task_name = 'hotpot.subqs-subas-q-a'
        elif task_name == 'hotpot-comparison-questions-cond-lm':
            task_name = 'hotpot.q-subqs.comparison'

        filename = args.context_path.split('/')[-1]
        split = None
        for possible_split in ['train', 'dev', 'test']:
            if possible_split in filename:
                split = possible_split
                break
        assert split is not None, 'Unable to determine split for context_path {}'.format(args.context_path)

        with open(args.context_path, 'r') as f:
            all_data = json.load(f)

        # Filter for comparison questions only
        examples = []  # Need to fill in this variable to add prompts
        if task_name == 'hotpot.q-subqs.comparison':
            save_data = {'data': []}
            for i, example in enumerate(all_data['data']):
                if (i % args.num_shards) != args.shard_no:
                    continue
                assert len(example['paragraphs']) == 1, 'Unexpected # paragraphs: {}'.format(len(example['paragraphs']))
                assert len(example['paragraphs'][0]['qas']) == 1, 'Unexpected # paragraphs: {}'.format(
                    len(example['paragraphs'][0]['qas']))
                if example['paragraphs'][0]['qas'][0]['type'] == 'comparison':
                    example['paragraphs'][0]['qas_orig'] = example['paragraphs'][0]['qas']
                    example['paragraphs'][0]['qas'] = []
                    example['prompt'] = ['paragraphs'][0]['qas_orig'][0]['question']
                    if format_tokenizer:
                        example['prompt'] = format_tokenizer.decode(
                            format_tokenizer.encode(example['prompt'].strip())).strip()
                    example['prompt'] = q_sep + ' ' + example['prompt'] + ' ' + q_sep
                    save_data['data'].append(example)
                    examples.append(example['prompt'])
        elif task_name in {'hotpot.subqs-subas-q-a', 'hotpot.q-sfs-a', 'squad.sf-q'}:
            if task_name == 'squad.sf-q':
                eos_sep = a_sep
                eoi_sep = q_sep
            else:
                eos_sep = q_sep
                eoi_sep = a_sep

            examples_with_answers = load_dataset(split, task_name)
            answers = []
            stats_sum = {'em': 0, 'f1': 0}
            for example_with_answer in examples_with_answers:
                example, answer = example_with_answer.rsplit(eoi_sep, 1)
                examples.append(example + eoi_sep)
                answers.append(answer.strip(eos_sep).strip())
        else:
            raise NotImplementedError('task_name {}'.format(task_name))

    tqdm_bar = trange(1000 if task_name is None else len(examples), desc='Evaluating')
    for d in tqdm_bar:
        generated = 0
        context_tokens = None
        start_token = enc.encoder['<|endoftext|>']
        out_start_index = 1
        if not args.unconditional:
            if task_name is None:
                raw_text = input("Model prompt >>> ")
                while not raw_text:
                    print('Prompt should not be empty!')
                    raw_text = input("Model prompt >>> ")
                if format_tokenizer:
                    raw_text = format_tokenizer.decode(format_tokenizer.encode(raw_text.strip())).strip()
            else:
                raw_text = examples[d]
                if task_name == 'hotpot.q-subqs.comparison':
                    assert save_data['data'][d]['prompt'] == examples[d]

            context_tokens = enc.encode(raw_text)
            start_token = None
            out_start_index = len(context_tokens)

        for _ in range(args.nsamples // args.batch_size):
            out = sample_sequence(
                model=model, length=args.length, task_name=task_name, context=context_tokens, start_token=start_token,
                batch_size=args.batch_size, temperature=args.temperature, top_k=args.top_k, device=device,
                end_token=None if task_name is None else enc.encode(' ' + eos_sep)[0]
            )
            out = out[:, out_start_index:].tolist()

        for i in range(args.batch_size):
            generated += 1
            text = enc.decode(out[i])
            print(text)
            if task_name == 'hotpot.q-subqs.comparison':
                text = text.strip(q_sep)  # Remove EOS token
                subqs = text.split(q_sep)  # Split apart generated sub-Qs TODO: + operator (if applicable)
                # assert len(subqs) == 2, 'Unexpected # of generated subqs: {}'.format(len(subqs) == 2)  # TODO: Add
                for subq_no, subq_text in enumerate(subqs[:-1]):
                    subq_text = subq_text.strip()
                    save_data['data'][d]['paragraphs'][0]['qas'].append({
                        # 'final_answers': [],
                        'question': subq_text,
                        'level': 'subquestion',
                        'type': 'subquestion' + str(subq_no),
                        'id': save_data['data'][d]['paragraphs'][0]['qas_orig'][0]['id'] + '-' + str(subq_no),
                        'answers': [[] for _ in range(len(save_data['data'][d]['paragraphs'][0]['context']))],
                    })
                save_data['data'][d]['paragraphs'][0]['op'] = subqs[-1].replace(' ', '_').upper()
            elif task_name in {'hotpot.subqs-subas-q-a', 'hotpot.q-sfs-a', 'squad.sf-q'}:
                # EM
                pred_answer = clean_up_tokenization_spaces(text.split(eos_sep)[0].strip())
                answer = clean_up_tokenization_spaces(answers[d])
                em = pred_answer == answer
                stats_sum['em'] += em

                # F1
                pred_answer_bow = set(pred_answer.split())
                answer_bow = set(answer.split())
                precision = len(pred_answer_bow.intersection(answer_bow)) / len(pred_answer_bow)
                recall = len(pred_answer_bow.intersection(answer_bow)) / len(answer_bow)
                f1 = stats.hmean([precision, recall]) if ((precision != 0) and (recall != 0)) else 0
                stats_sum['f1'] += f1

                print(raw_text)
                print(int(em), round(100 * f1), pred_answer, '/', answers[d])
                tqdm_bar.desc = "Evaluation EM: {:.1%} F1: {:.1%}".format(stats_sum['em'] / (d + 1),
                                                                          stats_sum['f1'] / (d + 1))

    # Print or save results
    if task_name == 'hotpot.q-subqs.comparison':
        with open('data/decomposed-predictions/comparison_decomposed_{}_generations.num_shards={}.shard_no={}.json'.format(split, args.num_shards, args.shard_no), 'w') as f:
            json.dump(save_data, f)
    elif task_name in {'hotpot.subqs-subas-q-a', 'hotpot.q-sfs-a'}:
        print("Evaluation EM: {:.1%} F1: {:.1%}".format(stats_sum['em'] / len(examples),
                                                        stats_sum['f1'] / len(examples)))
    print('Completed in {:.0f}s'.format(time.time() - start_time))


if __name__ == '__main__':
    run_model()


