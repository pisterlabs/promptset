import json
import re
import openai
import time

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--input_file', required=True)
parser.add_argument('--save_to', required=True)
parser.add_argument('--shots_file')
parser.add_argument('--pseudo_file')
parser.add_argument('--dataset')
parser.add_argument('--train_mode')

args = parser.parse_args()

data = []
with open(args.input_file) as f:
    for line in f.readlines():
        data.append(json.loads(line))

if args.shots_file and args.pseudo_file:
    shots = []
    with open(args.shots_file) as f:
        for line in f.readlines():
            shots.append(json.loads(line))

    cands = {}
    with open(args.pseudo_file) as f:
        for line in f.readlines():
            line = json.loads(line)
            cands[line['uid']] = line['cat']

openai.api_key = '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'

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
            time.sleep(1)
    if return_text:
        completion = completion['choices'][0]['message']['content']
    return completion


def clean_new(data):
    count = 0
    for i, d in enumerate(data):
        preds = d['pred']
        # if len(d['content']) > 10000:
        #     # count += 1
        #     prompt = "Product title: " + d['title'] + '\nPlease predict at least 10 other products titles. \n Format: ["title1", "title2", "title3", "title4", "title5"], do not say any word or explain. \n'
        #     preds = get_completion_with_retries(prompt)
        #     try:
        #         preds = json.loads(preds)
        #     except Exception as e:
        #         pass
        # else:
        #     preds = d['pred']
        # with open('tmp/origin.txt', 'a') as f:
        #     f.write(preds + '\n\n')
        preds = [pred for pred in preds.lower().strip().split('\n') if pred != '']
        if len(preds) == 1:
            count += 1
            if args.train_mode == 'ez':
                shots_prompt = ''
                for j in range(min(len(shots[i]), 5)):
                    shot_title = shots[i][j]['title']
                    if args.dataset == 'amazon':
                        shot_cands = '\n'.join([item['text'] for item in cands[shots[i][j]['uid']][:5]])
                        shot_prompt = "Product title on Amazon: " + shot_title + "\nSimilar product: " + shot_cands + '\n'
                    if args.dataset == 'wiki':
                        shot_cands = '\n'.join(cands[shots[i][j]['uid']][:5])
                        shot_prompt = "Passage title on Wikipedia: " + shot_title + "\nSimilar passage: " + shot_cands + '\n'
                shots_prompt += shot_prompt
                prompt = "You are now trying to predict at least 10 relevant passages for a new Wikipedia passage title: " + d['title'] + "\nOnly output titles with line break, do not include anything else. example: title 1\ntitle2\n..."
                preds = get_completion_with_retries(prompt)
                preds = [pred for pred in preds.lower().strip().split('\n') if pred != '']
            if len(preds) == 1:
                preds = [x.strip() for x in re.split(r'(?<!\s),(?!\s)', preds[0])]
                if len(preds) == 1 and len(preds[0].split(',')) >= 5:
                    preds = preds[0].split(',')
                preds = [pred for pred in preds if pred != '']
                print(len(preds))
                # print(preds)

        # with open('tmp/final.txt', 'a') as f:
        #     f.write(json.dumps(preds) + '\n')
        d['pred'] = preds
    print(count)
    return data

data = clean_new(data)

with open(args.save_to, "w") as outfile:
    for d in data:
        outfile.write(
            json.dumps(
                {
                    "id": d['uid'],
                    "output": d['pred']
                }
            ) + "\n"
        )