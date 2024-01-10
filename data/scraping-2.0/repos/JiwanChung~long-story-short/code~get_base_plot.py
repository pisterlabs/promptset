import os
import json
from pathlib import Path

import openai
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from transformers import AutoTokenizer

from args import get_args
from loader.base import DataSpecifics, LoadSubtt
from utils import run_with_tries, format_context


def main():
    args = get_args()
    dset = args.dataset
    data_path = f'../../data/{dset}'
    data_path = Path(data_path)
    get_specifics = DataSpecifics(args, data_path, dset)
    load_subtt = LoadSubtt(args, data_path, dset)
    run(args, data_path, get_specifics, load_subtt)


def run_row(pargs):
    start_sequence = "\nA:"

    args, fixed_prompt, tokenizer, out_dir, vid_path, get_starter, load_subtt, key = pargs
    out_path = out_dir / (key + '.split.wiki')
    if out_path.exists() and not args.debug:
        print("exist " + key)
        with open(out_path) as f:
            res = json.load(f)
        return (key, res)
    keys, chunks, context = load_subtt(vid_path, key)

    prev_plots = []
    ok_keys = []
    prev = ""
    for k, c_ in tqdm(zip(keys, chunks), total=len(chunks), desc='chunks'):
        gap = '\n\n'
        subtt = format_context(c_, gap)
        if len(subtt) == 0:
            continue
        curr_prompt = '\n\nSynopsis:'

        if context is not None:
            subtt = context + gap + subtt

        if (len(subtt) > 3500):
            subtt = subtt[:3500]

        session_prompt = fixed_prompt + gap + subtt + curr_prompt
        if args.prev_prompt_num > 0:
            prevs = prev_plots[-args.prev_prompt_num:]
            if len(prevs) > 0:
                prev_prompt = '\n\nPrevious synopsis:\n'
                prev_prompt = prev_prompt + ''.join(prevs).strip() + '\n'
                session_prompt = fixed_prompt + gap + subtt + prev_prompt + curr_prompt

        response = run_with_tries(args, session_prompt, max_tokens=args.max_plot_tokens)

        prev = response['choices'][0]['text'].strip()
        prev_plots.append(prev)
        ok_keys.append(k)

    res = dict(zip(ok_keys, prev_plots))
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=4)
    return (key, res)


def run(args, data_path, get_specifics, load_subtt):
    '''
    if args.dataset == 'dramaqa':
        args.prev_prompt_num = 1
    '''
    fixed_prompt = "I am a highly intelligent storytelling bot. If you give me a subtitle, I will give you the short synopsis in detail.\n"# in a couple of sentences. \n"

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.model_max_length = 4000

    out_dir = data_path.parent / 'outputs' / args.dataset / 'summary' / 'base_plot' / args.name
    if args.debug:
        out_dir = data_path.parent / 'outputs' / 'debug' / 'summary' / 'base_plot' / args.name / args.dataset
    out_dir.mkdir(exist_ok=True, parents=True)

    openai.api_key = args.api_key

    vid_path, vids, get_starter, prompts = get_specifics()
    if 'base_plot' in prompts:
        fixed_prompt = prompts['base_plot']

    if args.debug:
        if args.dataset == 'movieqa':
            vids = ['tt0105414']
        elif args.dataset == 'dramaqa':
            vids = ['AnotherMissOh07_009']
        else:
            vids = vids[:1]

    pargs = [(args, fixed_prompt, tokenizer, out_dir, vid_path, get_starter, load_subtt, key)
             for key in vids]
    if args.do_mp:
        res = list(process_map(run_row, pargs, max_workers=args.num_workers, chunksize=10))
    else:
        res = list(tqdm(map(run_row, pargs), total=len(pargs)))
    return res


if __name__ == '__main__':
    main()
