import os
import argparse
import openai
import tiktoken

import json
import time
from tqdm.auto import tqdm
from nltk.corpus import stopwords


parser = argparse.ArgumentParser()
parser.add_argument('--target_model', type=str, default='text-davinci-003')
parser.add_argument('--dataset_name', type=str, default='LAMA_TREx')
parser.add_argument('--dataset_type', type=str, default='test')
args = parser.parse_args()

openai.api_key = os.getenv("OPENAI_API_KEY")

encoding = tiktoken.encoding_for_model(args.target_model)

stopword_list = stopwords.words("english")
stopword_ids = []
for stopword in stopword_list:
    token_ids = encoding.encode(' '+stopword)
    if len(token_ids) == 1:
        stopword_ids.append(token_ids[0])

logit_bias_remove_stopwords = {}
for stopword_id in stopword_ids:
    logit_bias_remove_stopwords[str(stopword_id)] = -100

with open(f'data/{args.dataset_name}/{args.dataset_type}.json') as fin:
    test_data = json.load(fin)

uids = []
prompts = []

for example in tqdm(test_data):
    uid = example['uid']
    prompt = example['truncated_input']

    uids.append(uid)
    prompts.append(prompt)

raw_predictions = []
raw_predictions_remove_stopwords = []

batch_size = 100
for i in tqdm(range(0, len(prompts), batch_size)):
    uid_batch = uids[i:i+batch_size]
    prompt_batch = prompts[i:i+batch_size]

    while True:
        try:
            responses = openai.Completion.create(
                model=args.target_model,
                prompt=prompt_batch,
                max_tokens=1,
                temperature=0,
                logprobs=5,
            )

            responses_remove_stopwords = openai.Completion.create(
                model=args.target_model,
                prompt=prompt_batch,
                max_tokens=1,
                temperature=0,
                logprobs=5,
                logit_bias=logit_bias_remove_stopwords,
            )

            break
        except Exception as e:
            print(e)
            time.sleep(3)

    for uid, response in zip(uid_batch, responses.choices):
        raw_predictions.append({"uid": uid, "response": response})
    for uid, response_remove_stopwords in zip(uid_batch, responses_remove_stopwords.choices):
        raw_predictions_remove_stopwords.append({"uid": uid, "response": response_remove_stopwords})


out_path = os.path.join('results', args.target_model)
os.makedirs(out_path, exist_ok=True)

with open(os.path.join(out_path, f'raw_pred_{args.dataset_name}_{args.dataset_type}.json'), 'w') as fout:
    json.dump(raw_predictions, fout)

with open(os.path.join(out_path, f'raw_pred_{args.dataset_name}_{args.dataset_type}_remove_stopwords.json'), 'w') as fout:
    json.dump(raw_predictions_remove_stopwords, fout)