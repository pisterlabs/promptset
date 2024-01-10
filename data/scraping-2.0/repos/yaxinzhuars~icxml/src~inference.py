import os
import json
import openai
import concurrent.futures
from tqdm import tqdm
import time
from openai.error import RateLimitError
from argparse import ArgumentParser
import random

openai.api_key = ''

parser = ArgumentParser()
parser.add_argument('--prompt_type', required=True, choices=['weak', 'full', 'example'])
parser.add_argument('--weak_file')
parser.add_argument('--lbl_file')
parser.add_argument('--test_file')
parser.add_argument('--pseudo_file')
parser.add_argument('--shots_file')
# parser.add_argument('--input_file', required=True)
parser.add_argument('--save_to', required=True)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=1000000)
parser.add_argument('--dataset', required=True, choices=['amazon', 'wiki'])
args = parser.parse_args()

def get_completion_with_retries(prompt, return_text=True, reduce_length=False, tqdm=None):
    while True:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                timeout=30
            )
            break
        except Exception as e:
            print(str(e))
            if "This model's maximum context length is" in str(e):
                print('reduce_length')
                return 'ERROR::reduce_length'
            # self.key_id = (self.key_id + 1) % len(self.key)
            # openai.api_key = self.key[self.key_id]
            time.sleep(5)
    if return_text:
        completion = completion['choices'][0]['message']['content']
    return completion


def main():
    lbl, text = [], []
    with open(args.lbl_file, encoding='latin-1') as f:
        for line in f.readlines():
            lbl.append(json.loads(line))
    with open(args.test_file) as f:
        for line in f.readlines():
            text.append(json.loads(line))
    shots = {}
    with open(args.shots_file) as f:
        for i, line in enumerate(f.readlines()):
            shots[i+args.start] = json.loads(line)

    random_prompt = ''
    with open('../preprocessing/random10prompt.txt') as f:
        for line in f.readlines():
            random_prompt += line

    cands = {}
    with open(args.pseudo_file) as f:
        for line in f.readlines():
            data = json.loads(line)
            cands[data['uid']] = data['cat']
            # print(data['uid'], data['cat'])
    

    prompts = []
    # for i in range(len(text)):
    for i in range(args.start, args.end):
        shots_prompt = ''
        for j in range(min(len(shots[i]), 10)):
            shot_title = shots[i][j]['title']
            if isinstance(shot_title, list):
                shot_title = shot_title[0]
            # shot_content = shots[i][j]['content']
            if args.prompt_type == 'full':
                shot_cands = '\n'.join([json.loads(item)['title'] for item in shots[i][j]['cat']])
            if args.prompt_type == 'example':
                shot_cands = '\n'.join(shots[i][j]['cat'])
            if args.prompt_type == 'weak':
                if args.dataset == 'wiki':
                    shot_cands = '\n'.join(cands[shots[i][j]['uid']][:10])
                else:
                    shot_cands = '\n'.join([item['text'] for item in cands[shots[i][j]['uid']][:5]])
            # shot_prompt = "Product title on Amazon: " + shot_title + ". Product content: " + shot_content + "\n Similar product: " + shot_cands + '\n'
            if args.dataset == 'amazon':
                # true
                # shot_prompt = "Product title on Amazon: " + shot_title + "\nSimilar product: " + shot_cands + '\n'
                # label
                _shot_cands = '\n'.join(x['title'] for x in random.sample(lbl, 5))
                shot_prompt = "Product title on Amazon: " + shot_title + "\nSimilar product: " + _shot_cands + '\n'
                # input
                # random_word = random.choice(temp)['title']
                # shot_prompt = "Product title on Amazon: " + random_word + "\nSimilar product: " + shot_cands + '\n'
            if args.dataset == 'wiki':
                print(shot_cands)
                print()
                print(shot_title)
                shot_prompt = "Title: " + shot_title + "\n'See Also' page: " + shot_cands + '\n'
            shots_prompt += shot_prompt
        
        if shots_prompt == '':
            shots_prompt = random_prompt

        title = text[i]['title']
        content = text[i]['content']
        if args.dataset == 'amazon':
            prompt = "You are now trying to predict at least 10 relevant products for a new Amazon product title: " + title + "\ndescription: " + content + '\n'
        elif args.dataset == 'wiki':
            # prompt = "You are now trying to predict at least 10 relevant passages for a new Wikipedia passage title: " + title + "\nOnly output titles with line break, do not include anything else. example: title 1\ntitle2\n...content: " + content[:1000] + '\n'
            prompt = "You are now trying to generate 'See also' suggestions related to the Wikipedia title: " + title + "\nOnly output titles with line break, do not include anything else. example: title 1\ntitle2\n...\nwiki content: " + content + '\n'
        # prompt_side = "background knowledge: " + background[i]['background'] + '\n'
        # hint = '\n'.join(hints[i]['output'])
        # hint = '(Hint: The answer may near to: ' + hint + ')'
        hint = ''
        prompt = shots_prompt + prompt
        print(prompt)
        print(len(prompt))
        prompt = prompt[:12000]
        prompts.append(prompt)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        completions = list(tqdm(executor.map(get_completion_with_retries, prompts), total=len(prompts)))
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     with tqdm(total=len(prompts)) as pbar:
    #         completions = list(executor.map(lambda prompt: get_completion_with_retries(prompt, tqdm=pbar), prompts))
    #         pbar.update()

    with open(args.save_to, 'w') as fw:
        for i, completion in enumerate(completions):
            gt = [lbl[j]['title'] for j in text[i]['target_ind']]
            result = text[i]
            result['gt'] = gt
            result['pred'] = completion
        
            json.dump(result, fw)
            fw.write('\n')





main()