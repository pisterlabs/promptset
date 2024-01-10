import os
import json
import openai
import concurrent.futures
from tqdm import tqdm
import time
from openai.error import RateLimitError
from bertopic import BERTopic
from argparse import ArgumentParser
from random import sample

parser = ArgumentParser()
parser.add_argument('--prompt_type', required=True, choices=['rerank', 'select'])
parser.add_argument('--dataset', required=True, choices=['amazon', 'wiki', 'eurlex'])
parser.add_argument('--topic_model')
parser.add_argument('--cluster_file')
parser.add_argument('--random_file')
parser.add_argument('--hint_file')
parser.add_argument('--input_file', required=True)
parser.add_argument('--save_to', required=True)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=1000000)

args = parser.parse_args()

openai.api_key = ''

def get_completion(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    time.sleep(15)
    return completion.choices[0].message.content



def get_completion_with_retries(prompt, return_text=True, reduce_length=False, tqdm=None):
    while True:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4-0613",
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
            time.sleep(10)
    if return_text:
        completion = completion['choices'][0]['message']['content']
    return completion

def decode(start, end, text, lbl, preds, prompt_type, save_to):
    prompts = []
    # for i in range(len(text)):
    for i in range(start, end):
        title = text[i]['title']
        des = text[i]['content']
        cands = preds[i-start]['output']
        cands_prompt = ''
        # prompt_top10 = "The query product is: " + title + "\ndescription is: " + des + "\nHere are some candidate relevant products "
        # post_prompt = "Select top 10 products based on their relevance to the query product " + title
        
        for cid, cand in enumerate(cands):
            cand_prompt = '[' + str(cid + 1) + '] ' + cand + '\n'
            cands_prompt += cand_prompt
        
        # prompt = prompt + cands_prompt + 'The ranking results of the ' + str(len(cands)) + ' passages (only identifiers) is:'
        if args.dataset == 'amazon':
            prompt_select = "**Task**: Given a query product, select the top 10 most relevant products from a list of candidates.\n**Query product title**: " \
            + title + "\n**Candidates**:\n" + cands_prompt + "\n**Output format**: A list of integers representing the indices of the top 10 most relevant products. Example: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] \nOnly ouput the list, do not include any description or explanation. \n**Query Product description**: " + des 
        if args.dataset == 'wiki':
            # prompt_select = "**Task**: From the following candidate list of Wikipedia pages, select top 10 that would be most relevant for the 'See also' section of the given page:\n**wiki title**: " \
            # + title + "\n**Candidates**:\n" + cands_prompt + "\n**Output format**: A list of integers representing the indices of the top 10 most possible titles. Example: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] \nOnly ouput the list, do not include any description or explanation. \n**wiki content**: " + des 
            prompt_select = "**Task**: From the following candidate list of Wikipedia pages, select top 10 that would be most relevant for the 'See also' section of the given page:\n**wiki title**: " \
            + title + "\n**Output format**: A list of integers representing the indices of the top 10 most possible titles. Example: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] \nOnly ouput the list, do not include any description or explanation. \n**Candidates**:\n" + cands_prompt + '\n**wiki content**: ' + des 
        if args.dataset == 'eurlex':
            prompt_select = "**Task**: From the following candidate list of labels, select top 10 that would be most relevant for the EU legislative document:\n**doc title**: " \
            + title + "\n**Output format**: A list of integers representing the indices of the top 10 most possible labels. Example: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] \nOnly ouput the list, do not include any description or explanation. \n**Candidates**:\n" + cands_prompt + '\n**doc header**: ' + text[i]['header'] \
            + "\n**doc recitals**: " + text[i]['recitals'] 
    
        prompt = prompt_select

        print(prompt)
        # print(len(prompt))
        prompt = prompt[:12000]
        prompts.append(prompt)
        # print(prompt)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        completions = list(tqdm(executor.map(get_completion_with_retries, prompts), total=len(prompts)))
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     with tqdm(total=len(prompts)) as pbar:
    #         completions = list(executor.map(lambda prompt: get_completion_with_retries(prompt, tqdm=pbar), prompts))
    #         pbar.update()

    with open(save_to, 'w') as fw:
        for i, completion in enumerate(completions):
            gt = [lbl[j]['title'] for j in text[i+start]['target_ind']]
            result = text[i+start]
            result['gt'] = gt
            result['pred'] = completion
            json.dump(result, fw)
            fw.write('\n')

def main():
    lbl, text = [], []
    # results = json.load(open('/work/yaxinzhu_umass_edu/chatgpt/xml/AmazonCat-13K.bow/tstfidf.json'))
    if args.dataset == 'amazon':
        dataset = 'LF-Amazon-131K'
    if args.dataset == 'wiki':
        dataset = 'LF-WikiSeeAlso-320K'
    if args.dataset == 'eurlex':
        dataset = 'EURLex-4.3K'
    with open('../xml/' + dataset + '/lbl.json', encoding='latin-1') as f:
        for line in f.readlines():
            lbl.append(json.loads(line))
    with open('../xml/' + dataset + '/tst.json') as f:
        for line in f.readlines():
            text.append(json.loads(line))

    # shots = []
    # with open('../preprocessing/train_title_10shot.txt') as f:
    #     for line in f.readlines():
    #         shots.append(json.loads(line))

    preds = []
    with open(args.input_file) as f:
        for line in f.readlines():
            preds.append(json.loads(line))

    # topic_model = BERTopic.load(args.topic_model)
    # cluster_examples = json.load(open(args.cluster_file))
    topic_model, cluster_examples = None, None
    random_prompt = ''
    if args.random_file:
        with open(args.random_file) as f:
            for line in f.readlines():
                random_prompt += line

    start, end = 0, len(text)
    if args.start is not None and args.end is not None:
        start = args.start
        end = args.end
    decode(start, end, text, lbl, preds, args.prompt_type, args.save_to)

main()
