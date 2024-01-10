# Copyright (C) 2023 ByteDance. All Rights Reserved.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import openai
import sys
from tqdm import tqdm
import json
import re
import random
import os
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from utils import chat_completion_with_backoff, completion_with_backoff,\
                  query_opt, query_flan_t5

openai.api_key = ''  # fill your own api_key.

MODELS = ['gpt-3.5-turbo', 'text-davinci-003',
          'davinci', 'gpt-4', 'opt-1.3b', 'flan-t5-xxl']


def openai_query(model, queries, temp=0.7):
    if model in ['gpt-3.5-turbo', 'gpt-4']:
        response = chat_completion_with_backoff(
            model=model,
            messages=[
                {"role": "user", "content": queries},
            ],
            temperature=temp, max_tokens=50)
        answer = response['choices'][0]['message']['content']
    elif model == 'text-davinci-003':
        response = completion_with_backoff(
            engine='text-davinci-003', prompt=queries,
            temperature=temp, max_tokens=50, logprobs=1)
        answer = response['choices'][0]['text']

    elif model == 'davinci':
        # Need a special QA format for the original GPT-3 model.
        response = completion_with_backoff(
            engine='davinci', prompt=queries,
            temperature=temp, max_tokens=50, logprobs=1)
        answer = response['choices'][0]['text']

    else:
        raise AssertionError

    answer = answer.strip()
    return answer


if __name__ == '__main__':
    data_path, model_ind = sys.argv[1], int(sys.argv[2])

    print('data_path: ', data_path, 'model_ind', model_ind)
    prompt = "Select the correct answer to the question, based on the provided knowledge.  Output (A) or (B) and explain why."

    if not os.path.exists('./eval_results_qa'):
        os.makedirs('./eval_results_qa')
    wf = open('./eval_results_qa/eval_%s.txt' % (MODELS[model_ind]), 'w')

    # load data
    model = MODELS[model_ind]
    print('testing: ', model)

    # OPT-1.3B.
    if model == 'opt-1.3b':
        opt_generator = pipeline('text-generation', model="facebook/opt-1.3b",
                                 do_sample=True, max_length=300, device="cuda:0")

    if model == 'flan-t5-xxl':
        # flan-t5-xxl.
        flan_t5_model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-xxl", device_map="auto")
        flan_t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")

    stats = {model: 0, 'count': 0, 'valid': 0}
    count = 0
    with open(data_path, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

    for js in tqdm(data):
        count += 1
        if count > 300:
            break
        knowledge = js["knowledge"]
        question = js["question"]
        hallucinated_answer = js["hallucinated_answer"]
        right_answer = js["right_answer"]

        wf.write('==========================\n')
        if random.random() > 0.5:
            an_A, an_B = right_answer, hallucinated_answer
            groundtruth = 'A'
        else:
            an_A, an_B = hallucinated_answer, right_answer
            groundtruth = 'B'

        queries = prompt + "\nKnowledge:"+knowledge + "\nQuestion:" + \
            question + "\n(A)" + an_A + "\n(B)" + an_B + '\nAnswer:'

        wf.write(queries+'\n')

        if model == 'opt-1.3b':
            response = query_opt(queries, opt_generator)
        elif model == 'flan-t5-xxl':
            response = query_flan_t5(
                queries, flan_t5_model, flan_t5_tokenizer)
        else:
            response = openai_query(model, queries=queries)

        wf.write('**response: %s \n --------\n' % (response))
        wf.write('**groundtruth: %s' % groundtruth + '\n')

        stats['count'] += 1

        stripped_response = response.strip(' ()')
        if len(stripped_response) > 0:
            responsed_option = stripped_response[0]
        else:
            responsed_option = 'EMPTY'
        if responsed_option not in ['A', 'B']:
            wf.write('##################' + responsed_option)
        else:
            stats['valid'] += 1

        if model == 'davinci':

            # davinci cannot generate well-formulated answer, double check
            if responsed_option not in ['A', 'B']:
                temp = re.split(r'\banswer is\b', response)
                if len(temp) > 1:
                    ss = temp[1].strip(' ()')
                    responsed_option = ss[0]

                    wf.write('--ss: '+ss+', extracted_option:%s\n' %
                             responsed_option)

            if responsed_option == groundtruth:
                stats[model] += 1

        else:
            if groundtruth == responsed_option:
                stats[model] += 1

        wf.write(json.dumps(stats)+'\n')

    print('stats: ', stats)
    wf.close()
