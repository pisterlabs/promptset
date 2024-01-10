import argparse
import json
import os

import openai
import tqdm
import ray
import time
from tqdm.contrib import tzip
# openai key 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
NUM_SECONDS_TO_SLEEP = 0

@ray.remote(num_cpus=4)
def get_eval(content: str, max_tokens: int):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model='gpt-35-turbo',
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
                }, {
                    'role': 'user',
                    'content': content,
                }],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=max_tokens,
            )
            break
        except openai.error.RateLimitError:
            pass
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    print('success!')
    return response['choices'][0]['message']['content']


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-q', '--question')
    # parser.add_argument('-a', '--answer')
    parser.add_argument('-a', '--answer-list', nargs='+', default=[])
    parser.add_argument('-r', '--rule')
    parser.add_argument('-o', '--output')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    ray.init()

    f_q_raw = open(os.path.expanduser(args.question))
    f_ans1_raw = open(os.path.expanduser(args.answer_list[0]))
    f_ans2_raw = open(os.path.expanduser(args.answer_list[1]))
    rule_dict = json.load(open(os.path.expanduser(args.rule), 'r'))

    review_file = open(f'{args.output}', 'w')

    f_q = []
    f_ans1 = []
    f_ans2 = []

    for line in f_q_raw:
        json_obj = json.loads(line)
        f_q.append(json_obj)

    for line in f_ans1_raw:
        json_obj = json.loads(line)
        f_ans1.append(json_obj)
    
    for line in f_ans2_raw:
        json_obj = json.loads(line)
        f_ans2.append(json_obj)


    assert len(f_q) == len(f_ans1) == len(f_ans2), 'The number of questions and answers are not equal.'
    assert len(f_q) > 0, 'The number of questions is 0.'
    

    js_list = []
    handles = []
    idx = 0
    for ques_js, ans1_js, ans2_js in tqdm.tqdm(tzip(f_q, f_ans1, f_ans2)):
        # if idx == 1:
        #     break

        ques = ques_js
        ans1 = ans1_js
        ans2 = ans2_js

        category = ques['category']
        if category in rule_dict:
            rule = rule_dict[category]
        else:
            rule = rule_dict['default']
        prompt = rule['prompt']
        role = rule['role']
        content = (f'[Question]\n{ques["text"]}\n\n'
                   f'[{role} 1]\n{ans1["text"]}\n\n[End of {role} 1]\n\n'
                   f'[{role} 2]\n{ans2["text"]}\n\n[End of {role} 2]\n\n'
                   f'[System]\n{prompt}\n\n')
        js_list.append({
            'id': idx+1,
            'question_id': ques['question_id'],
            'answer1_id': ans1['question_id'],
            'answer2_id': ans2['answer_id'],
            'category': category})
        idx += 1
        handles.append(get_eval.remote(content, args.max_tokens))
        # To avoid the rate limit set by OpenAI
        time.sleep(NUM_SECONDS_TO_SLEEP)

    reviews = ray.get(handles)
    for idx, review in enumerate(reviews):
        scores = parse_score(review)
        js_list[idx]['content'] = review
        js_list[idx]['tuple'] = scores
        review_file.write(json.dumps(js_list[idx]) + '\n')
    review_file.close()

