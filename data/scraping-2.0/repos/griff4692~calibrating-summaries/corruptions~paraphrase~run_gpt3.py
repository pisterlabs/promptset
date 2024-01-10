import itertools
import regex as re
import os
from time import sleep
CWD = os.path.dirname(__file__)
from glob import glob

import openai
import argparse
import pandas as pd
from tqdm import tqdm
from random import choice

openai.api_key = os.getenv('OPENAI_API_KEY')
print(openai.api_key)
if openai.api_key is None:
    print('Need to set env var OPENAI_API_KEY to be able to call GPT-3')

from preprocess.preprocess import data_loader


def build_prompt(abstract, annotated_abstracts, summary_name='abstract'):
    prompt = f'Paraphrase this {summary_name}.\n\n'
    for orig, human in annotated_abstracts:
        prompt += orig.strip() + '=>' + choice(human).strip() + '\n\n'
    prompt += abstract + '=>'
    return prompt


def merge_chunks(output_dir, save_dir):
    out_fn = os.path.join(save_dir, 'predictions.csv')
    chunk_pattern = os.path.join(output_dir, '*.csv')
    print(f'Searching for files matching {chunk_pattern}...')
    chunk_fns = list(glob(chunk_pattern))
    print(f'Found {len(chunk_fns)} matching files')
    merged = []
    for fn in tqdm(chunk_fns):
        try:
            merged.append(pd.read_csv(fn))
        except:
            print(f'Could not parse file {fn}')
    merged = pd.concat(merged)
    print(f'Saving {len(merged)} outputs to {out_fn}')
    merged.sort_values(by='uuid').reset_index(drop=True).to_csv(out_fn, index=False)


def clean_uuid(uuid):
    clean = re.sub(r'\W+', '_', uuid)
    return re.sub(r'_+', '_', clean).strip('_')


def is_incomplete(record, out_dir):
    uuid = record['uuid']
    uuid_clean = clean_uuid(uuid)
    out_fn = os.path.join(out_dir, f'{uuid_clean}.csv')
    return not os.path.exists(out_fn)


def paraphrase_with_gpt(args, record, annotated_abstracts, summary_name):
    few_shot_examples = list(itertools.combinations(list(range(len(annotated_abstracts))), args.few_shot_n))
    sampled_example_set = [annotated_abstracts[i] for i in few_shot_examples[choice(range(len(few_shot_examples)))]]
    prompt = build_prompt(record['target'], sampled_example_set, summary_name=summary_name)

    response = openai.Completion.create(
        model='text-davinci-002',
        prompt=prompt,
        temperature=0.7,
        max_tokens=args.max_tokens,
        top_p=1,
        n=args.num_candidates,
        frequency_penalty=0,
        presence_penalty=0
    )

    return [x['text'] for x in response['choices']]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments to process extract entities')
    parser.add_argument('--data_dir', default=os.path.expanduser('~/data_tmp'))
    parser.add_argument('--dataset', default='pubmed', choices=['pubmed', 'clinical', 'chemistry'])
    parser.add_argument('-overwrite', default=False, action='store_true')
    parser.add_argument('--few_shot_n', default=1, type=int)
    parser.add_argument('--num_candidates', default=5, type=int)
    parser.add_argument('--max_tokens', default=512, type=int)
    parser.add_argument('--mode', default='generate', choices=['generate', 'merge'])

    args = parser.parse_args()

    save_dir = os.path.join(args.data_dir, args.dataset, 'paraphrase')
    out_dir = os.path.join(args.data_dir, args.dataset, 'paraphrase', 'gpt')
    os.makedirs(out_dir, exist_ok=True)

    if args.mode == 'merge':
        merge_chunks(out_dir, save_dir)
        exit(0)

    annotations_fn = os.path.join(args.data_dir, args.dataset, 'paraphrase', 'annotations.txt')
    with open(annotations_fn, 'r') as fd:
        paraphrase_annotations = fd.readlines()
        paraphrase_annotations = [x.strip() for x in paraphrase_annotations if len(x.strip()) > 0]

    paraphrase_annotation_tuples = []
    summary_name = 'Summary' if args.dataset == 'clinical' else 'Abstract'
    for idx in range(len(paraphrase_annotations)):
        if paraphrase_annotations[idx].startswith(f'{summary_name}:'):
            paraphrase_annotation_tuples.append([paraphrase_annotations[idx].replace(f'{summary_name}:', ''), []])
        else:
            assert 'Paraphrase:' in paraphrase_annotations[idx]
            paraphrase_annotation_tuples[-1][1].append(paraphrase_annotations[idx].replace('Paraphrase:', ''))

    dataset = data_loader(args.dataset, contrast_subsample=True)

    for split, data in dataset.items():
        prev_n = len(data)
        if not args.overwrite:
            print('Filtering out already done examples...')
            data = list(filter(lambda x: is_incomplete(x, out_dir), data))

        for record in tqdm(data):
            uuid = record['uuid']
            uuid_clean = clean_uuid(uuid)
            out_fn = os.path.join(out_dir, f'{uuid_clean}.csv')
            try:
                paraphrases = paraphrase_with_gpt(args, record, paraphrase_annotation_tuples, summary_name)
            except openai.error.RateLimitError:
                print('Rate limit exceeded. Sleeping for a minute and re-trying.')
                sleep(60)
                paraphrases = paraphrase_with_gpt(args, record, paraphrase_annotation_tuples, summary_name)
            except openai.error.InvalidRequestError as e:
                print(e)
                print('Skipping for now.')
                continue

            output_df = pd.DataFrame([
                {
                    'uuid': record['uuid'], 'split': split, 'target': record['target'],
                    'prediction': p, 'paraphrase_idx': i
                } for i, p in enumerate(paraphrases)
            ])
            output_df.to_csv(out_fn, index=False)

    merge_chunks(out_dir, save_dir)
