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
import os

os.environ['TOKENIZERS_PARALLELISM']='true'

parser = ArgumentParser()
parser.add_argument('--prompt_type', required=True, choices=[
                                                             'zeroshot', 'example'])
parser.add_argument('--topic_model')
parser.add_argument('--cluster_file')
parser.add_argument('--random_file')
parser.add_argument('--hint_file')
# parser.add_argument('--input_file', required=True)
parser.add_argument('--save_to', required=True)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=1000000)
parser.add_argument('--dataset', choices=['wiki', 'amazon', 'eurlex'])

# parser.add_argument('--cluster_num')
args = parser.parse_args()

openai.api_key = ''

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
            time.sleep(60)
    if return_text:
        completion = completion['choices'][0]['message']['content']
    return completion


def decode(start, end, text, lbl, prompt_type, save_to):
    prompts = []
    # for i in range(len(text)):
    for i in range(start, end):
        title = text[i]['title']
        des = text[i]['content']
        # prompt = "You are now trying to predict relevant products for a new product title: " + title + '. Only output the product results, one each line, splitted with line break "\n", do not say any word or explain. \n'
        prompt = "You are now trying to predict at least 10 relevant products for a new product title: " + title + ': \n'

        if prompt_type == 'zeroshot':
            hint = '\n'.join(hint_prompt[i]['output'][:100])
            if hint != '':
                hint = '\n(Hint: The answer may near to: \n ' + hint + ')\n'
            # hint = ''
            # prompt = "Please predict at least 10 relevant products for an Amazon product title: " + title + "\ndescription: " + des + '\n'
            # prompt = "Given an Amazon product title: " + title + ", please generate a discription "
            if args.dataset == 'amazon':
                prompt = "Please predict at least 10 relevant products for an Amazon product title: " + title + hint + "product description: " + des + "\n"
            if args.dataset == 'wiki':
                prompt = "Generate 'See also' suggestions related to the Wikipedia title: " + title + '\nOnly output titles with line break, do not include anything else. example: title 1\ntitle2\n...' + hint + 'wiki_content: ' + des + '\n'
            if args.dataset == 'eurlex':
                prompt = "Generate tags/labels related to the EU legislative **document title**: " + title + '\nOnly output titles with line break, do not include anything else. example: title 1\ntitle2\n...' + '\n**document header**: ' + text[i]['header'] + '\n**document recitals**: ' + text[i]['recitals'] + '\n'
        elif prompt_type == 'example':
            if args.dataset == 'amazon':
                prompt = "Product title: " + title + '\nPlease predict at least 5 similar Amazon products titles. \n Format: ["title1", "title2", "title3", "title4", "title5"], do not say any word or explain. \n' + "\nproduct description: " + des 
            if args.dataset == 'wiki':
                prompt = "Wiki title: " + title + '\nPlease generate at least 5 relevant and diverse wiki page titles. \n Format: ["title1", "title2", "title3", "title4", "title5"], do not say any word or explain. \n' + "\nwiki content: " + des 
        print(prompt)
        print(len(prompt))
        prompt = prompt[:13000]
        prompts.append(prompt)

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
    background = []
    with open(args.background_file) as f:
        for line in f.readlines():
            background.append(json.loads(line))


    start, end = 0, len(text)
    if args.start is not None and args.end is not None:
        start = args.start
        end = args.end
    decode(start, end, text, lbl, args.prompt_type, args.save_to)

main()
