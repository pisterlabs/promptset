import os
import math
import re
import json
import random
import csv
from pathlib import Path
from itertools import chain, product
from collections import defaultdict

import openai
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from transformers import AutoTokenizer

from args import get_args
from loader.base import LoadSubtt
from loader.qa import DataSpecificsQA
from utils import run_with_tries, format_context_li, IndexFinder


'''
do not run the baselines for fast evaluation
'''

all_exp_keys = [
    'base_plot/gt',
    'base_plot/search',
    'gt_plot/gt',
    'gt_plot/search',
    'gt_plot/random',
    'gt_plot/none',
    'none/all',
    'none/none',
]
'''
all_exp_keys = [
    'base_plot/gt',
    # 'base_plot/search',
]
'''


def main():
    args = get_args()
    random.seed(0)
    dset = args.dataset
    data_path = f'../../data/{dset}'
    data_path = Path(data_path)
    get_specifics = DataSpecificsQA(args, data_path, dset)
    load_subtt = LoadSubtt(args, data_path, dset)
    run(args, data_path, get_specifics, load_subtt)


def get_exp_keys(args, keys):
    if args.exp_keys is None:
        return keys
    else:
        _keys = args.exp_keys.split(',')
        return list(set(_keys) & set(keys))


def run_row(pargs):
    gap = '\n\n'
    index_finder = IndexFinder()
    args, tokenizer, out_dir, vid_path, \
        unload_data, get_gt_index, load_subtt, get_plots, q_ = pargs
    qid, question, answers, answer, key = unload_data(q_)

    _all_exp_keys = get_exp_keys(args, all_exp_keys)
    if args.debug:
        _all_exp_keys = ['base_plot/all']

    out_path = out_dir / f'{qid}.json'
    if out_path.exists() and not args.debug:
        print(f"exists: {qid}")
        return None

    res = {}
    res['vis_key'] = key
    res['question'] = question
    res['candidates'] = answers
    res['answer'] = answer
    res['qid'] = qid

    base_plot = get_plots.load_plot('base', key)

    opts_prompt = {
        'base_plot': (None, base_plot),
    }
    opts_lookup = ['none', 'random', 'search']

    gt_plot = get_plots.load_plot('gt', key)
    if gt_plot is not None:
        opts_prompt['gt_plot'] = (None, gt_plot)
    gt_index = get_gt_index(q_)
    if gt_index is not None:
        opts_lookup.append('gt')

    hypos = {}
    jobs = []
    exp_keys = []
    for opt_prompt, opt_lookup in product(list(opts_prompt.keys()), opts_lookup):
        exp_key = f'{opt_prompt}/{opt_lookup}'
        if exp_key not in _all_exp_keys:
            continue

        story, plot = opts_prompt[opt_prompt]
        prompt, out = get_prompt(args, exp_key, qid, tokenizer, vid_path,
                                 get_gt_index, load_subtt,
                                 res, q_, story, plot, opt_lookup)
        hypos[exp_key] = out
        jobs.append(prompt)
        exp_keys.append(exp_key)
    exp_key = 'none/all'
    if exp_key in _all_exp_keys:
        prompt, out = get_prompt(args, exp_key, qid, tokenizer, vid_path,
                                get_gt_index, load_subtt,
                                res, q_, None, None, 'all')
        hypos[exp_key] = out
        jobs.append(prompt)
        exp_keys.append(exp_key)
    exp_key = 'none/none'
    if exp_key in _all_exp_keys:
        prompt, out = get_prompt(args, exp_key, qid, tokenizer, vid_path,
                                get_gt_index, load_subtt,
                                res, q_, None, None, None)
        hypos[exp_key] = out
        jobs.append(prompt)
        exp_keys.append(exp_key)

    outs = run_jobs(args, jobs)
    probs = [post_gen(qid, v) for v in outs]
    for exp_key, prob in zip(exp_keys, probs):
        hypos[exp_key]['likelihood'] = prob
    res['hypos'] = hypos

    with open(out_path, 'w') as f:
        json.dump(res, f, indent=4)
    return (qid, res)


def run_jobs(args, prompts):
    res = []
    for prompt in prompts:
        response = run_with_tries(args, prompt, max_tokens=1,
                                  stop=["\n"], logprobs=7)
        y = response['choices'][0]
        # y = str(int(re.findall(r'\d+', y)[0]))
        res.append(y)
    return res
    '''
    response = run_with_tries(args, prompts, max_tokens=1,
                                stop=["\n"], logprobs=7)
    return response['choices']
    '''


def post_gen(key, response):
    logprobs = response['logprobs']['top_logprobs'][0]
    try:
        probs = {i: math.exp(logprobs[str(i)]) for i in range(5)}
    except:
        print(f'missing keys in prob: {key}')
        print(logprobs)
        probs = {i: math.exp(logprobs[str(i)]) if str(i) in logprobs else 0 for i in range(5)}
    return probs


def load_plot(x, add_index: bool = True):
    x = {int(k): v for k, v in x.items()}
    keys = sorted([v for v in x.keys()])
    plot_li = [x[k] for k in keys]
    if add_index:
        plot = [f"({i}) {v.strip()}" for i, v in enumerate(plot_li)]
    else:
        plot = [f"- {v.strip()}" for i, v in enumerate(plot_li)]
    plot = '\n'.join(plot)
    plot = f'Plot:\n{plot}'
    return plot


class GetPlots:
    def __init__(self, base_plot_dir, gt_plots):
        self.base_plot_dir = base_plot_dir
        self.gt_plots = gt_plots

    def load_plot(self, name, key, add_index: bool = True):
        x = getattr(self, f'_load_{name}_plot')(key)
        return x
        # return load_plot(x, add_index)

    def _load_base_plot(self, key):
        plot_path = self.base_plot_dir / (key + '.split.wiki')
        with open(plot_path) as f:
            x = json.load(f)
        return x

    def _load_gt_plot(self, key):
        if self.gt_plots is not None and key in self.gt_plots:
            x = self.gt_plots[key]
            return x
        else:
            return None


def get_lookup(args, qid, exp_key, key, tokenizer, vid_path,
               get_gt_index, load_subtt,
               q_, story, que_, lookup_type: str = 'none'):
    res = {}
    res['index'] = [None]
    res['support'] = None

    keys, chunks, context = load_subtt(vid_path, key)
    subtt = dict(zip(keys, chunks))
    keys = sorted(list(subtt.keys()))
    chunks = [subtt[k] for k in keys]
    if lookup_type == 'none':
        chunks = [None]
    elif lookup_type == 'random':
        index = random.choice(list(range(len(chunks))))
        chunks = [chunks[index]]
        res['index'] = index
        res['support'] = None
    elif lookup_type == 'gt':
        index = get_gt_index(q_, (keys, chunks))
        if not isinstance(index, int):
            chunks = [chunks[idx] for idx in index]
        else:
            chunks = [chunks[index]]
        res['index'] = index
    elif lookup_type == 'search':
        start_sequence = "\nTop 1 Plot Index: ("

        lookup_prompt = "\nI am a highly intelligent question answering bot. "
        lookup_prompt += "If you provide me with a question, "
        lookup_prompt += "I will give you an index of the plot you should lookup to solve it.\n"

        gap = '\n\n'

        qas = gap + lookup_prompt + que_ + start_sequence
        qas_len = len(tokenizer.tokenize(qas))
        max_len = min(3900, 4000 - qas_len - args.max_qa_tokens)
        story = tokenizer.decode(tokenizer.encode(story)[:max_len])
        session_prompt = story + qas

        if len(tokenizer.tokenize(session_prompt)) >= 4090:
            print(f"Too long lookup prompt: {qid}/{exp_key}")

        response = run_with_tries(args, session_prompt, max_tokens=args.max_qa_tokens,
                                  stop=["\n"])
        output = response['choices'][0]['text']
        output = '(' + output
        try:
            # index = [int(v[1:-1]) for v in re.findall(r'\(\d+\)', output)][:2]
            index = [v[1:-1] for v in re.findall(r'\([\d|, ]+\)', output)]
            index = list(chain(*[v.split(',') for v in index]))
            index = [int(v.strip()) for v in index][:2]
        except Exception as e:
            print(e)
        _index = []
        for ind in index:
            if ind < len(chunks):
                _index.append(ind)
        index = _index
        if len(index) < 0:
            tqdm.write('no index generated')
            if len(output) >= 10:
                index = [index_finder(story, output)]
            index = [None]

        chunks = [chunks[i] if i is not None else None for i in index]
        res['index'] = index
        res['support'] = output
    elif lookup_type == 'all':
        chunks = chunks

    chunks = [v for v in chunks if v is not None]
    return res, chunks


def get_prompt(args, exp_key, qid, tokenizer, vid_path,
               get_gt_index, load_subtt,
               res, q_, story, plot, lookup_type: str = 'none'):
    qa_prompt = (
        "\nI am a highly intelligent plot question answering bot. "
        "If you ask me a question and candidates, I will give you the index of answer.\n"
    )
    # "If you ask me a question and candidates, I will give you the index of answer.\n"
    gap = '\n\n'

    out = {}
    question = res['question']
    key = res['vis_key']
    que_ = 'Q: ' + question
    if plot is None:
        lookup_plot = ''
        qa_plot = ''
    else:
        lookup_plot = load_plot(plot, add_index=True)
        qa_plot = load_plot(plot, add_index=True)

    if story is None:
        lookup_story = lookup_plot
        qa_story = qa_plot
    else:
        lookup_story = story + gap + lookup_plot
        qa_story = story + gap + qa_plot

    lookup_res, chunks = get_lookup(args, qid, exp_key, key, tokenizer, vid_path,
                                    get_gt_index, load_subtt,
                                    q_, lookup_story, que_, lookup_type)
    out['lookup'] = lookup_res

    answers = '\n'.join([f'({i})' + '. ' + x for i,x in enumerate(res['candidates'])])

    situation = '\nCandidate:\n' + answers
    situation = que_ + situation
    qa_start_sequence = "\nA:"
    if chunks is not None and len(chunks) > 0:
        subtt = format_context_li(chunks, gap)
        if len(subtt) > 0:
            qa_story = qa_story + gap + subtt

    qas = '\n[Story end]\n' + qa_prompt + situation + qa_start_sequence

    qas_len = len(tokenizer.tokenize(qas))
    max_len = min(3900, 4000 - qas_len - args.max_qa_tokens)
    qa_story = tokenizer.decode(tokenizer.encode(qa_story)[:max_len])

    session_prompt = qa_story + qas

    if exp_key == 'none/none':
        session_prompt = qa_prompt + situation + qa_start_sequence

    # get likelihoods
    lh_prompt = session_prompt + ' ('
    if len(tokenizer.tokenize(lh_prompt))  >= 4090:
        print(f"Too long a qa prompt: {qid}/{exp_key}")

    return lh_prompt, out


def run(args, data_path, get_specifics, load_subtt):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.model_max_length = 4000

    base_plot_dir = data_path.parent / 'outputs' / args.dataset / 'summary' / 'base_plot' / args.name

    out_dir = data_path.parent / 'outputs' / args.dataset / 'qa' / args.split / args.out_name
    if args.debug:
        out_dir = data_path.parent / 'outputs' / 'debug' / 'qa' / args.split / args.out_name / args.dataset
    out_dir.mkdir(exist_ok=True, parents=True)

    vid_path, valqa, gt_plots, get_vid_key, get_gt_index, unload_data = get_specifics()

    if args.debug:
        if args.dataset == 'dramaqa':
            vids = set([1239, 1231])
            valqa = [row for row in valqa if row['qid'] in vids]
            assert len(valqa) > 0
        else:
            valqa = valqa[:10]

    get_plots = GetPlots(base_plot_dir, gt_plots)

    openai.api_key = args.api_key

    pargs = [(args, tokenizer, out_dir, vid_path,
              unload_data, get_gt_index, load_subtt, get_plots, q_)
             for q_ in valqa]

    if args.do_mp:
        res = list(process_map(run_row, pargs, max_workers=args.num_workers, chunksize=10))
    else:
        res = list(tqdm(map(run_row, pargs), total=len(pargs)))
    print('done')
    return res


if __name__ == '__main__':
    main()
