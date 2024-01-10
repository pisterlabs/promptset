from argparse import ArgumentParser
import jsonlines
from tqdm import tqdm
import os
import openai
import backoff
import re


@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.Timeout, openai.error.APIError))
def get_response(hps, pmp):
    responds = openai.ChatCompletion.create(
        model=hps.model,
        messages=pmp,
        max_tokens=hps.max_tokens,
        temperature=hps.temperature,
        api_key=openai.api_key,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return responds


def get_prediction_explanation(task, response):
    text = response['choices'][0]['message']['content'].strip()
    try:
        temp, gpt4_pred = text.split("My Answer:")
        summary, summary_pred = temp.split("Summary Answer:")
        gpt4_pred = re.findall(pattern, gpt4_pred.replace("Answer", 'answer'))
        summary_pred = re.findall(pattern, summary_pred.replace("Answer", 'answer'))
        if len(gpt4_pred) >= 1:
            gpt4_pred = gpt4_pred[0]
        else:
            gpt4_pred = "None"
        
        if len(summary_pred) >= 1:
            summary_pred = summary_pred[0]
        else:
            summary_pred = "None"
    except:
        summary = text.strip()
        gpt4_pred = "None"
        summary_pred = "None"
    if task == "strategyqa":
        gpt4_pred = gpt4_pred.lower()
    
    return summary, gpt4_pred, summary_pred


def hyper_parameters():
    parser = ArgumentParser('Few-Shot')

    parser.add_argument('--candidate', type=str, default='./')
    parser.add_argument('--dataset', type=str, default='csqa ')
    parser.add_argument('--api', type=str, default='')
    parser.add_argument('--model', type=str, default='gpt-4')
    parser.add_argument('--mode', type=str, default='debate')

    parser.add_argument('--output_dir', type=str, default='./')
    parser.add_argument('--num_sequences', type=int, default=1)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.0)

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    hps = hyper_parameters()

    print('[HPS]: {}'.format(hps))
    print('[API Key]: {}'.format(hps.api))
    print('[Data]: {}'.format(hps.dataset))
    print('[Model]: {}'.format(hps.model))

    candidates = jsonlines.open(os.path.join(hps.candidate, '{}.jsonl'.format(hps.dataset)), 'r')
    candidates = [c for c in candidates]

    openai.api_key = hps.api

    questions = []
    if os.path.exists(os.path.join(hps.output_dir, '{}_summary.jsonl'.format(hps.dataset))):
        fi = jsonlines.open(os.path.join(hps.output_dir, '{}_summary.jsonl'.format(hps.dataset)), 'r')
        questions += [d['Q'] for d in fi]
        fi.close()
        fo = jsonlines.open(os.path.join(hps.output_dir, '{}_summary.jsonl'.format(hps.dataset)), 'a')
    else:
        fo = jsonlines.open(os.path.join(hps.output_dir, '{}_summary.jsonl'.format(hps.dataset)), 'w')
    
    if hps.dataset == 'strategyqa':
        instruction = "You are given a yes/no Question. There is a debate on this question between user1 and user2, one user might give in, please summarise the debate very shortly. Then give two kinds of answers, one is based on the debate process, and the other is based on your own knowledge. Your response should be in the format like \"Summary: ___\nSummary Answer: (yes or no) is more plausible.\nMy Answer: (yes or no) is more plausible.\" Remember that you should choose only one option for each answer."
    else:
        instruction = "You are given a Question and its corresponding Options. There is a debate on this question between user1 and user2, one user might give in, please summarise the debate very shortly. Then give two kinds of answers, one is based on the debate process, and the other is based on your own knowledge. Your response should be in the format like \"Summary: ___\nSummary Answer: (A or B) is more plausible.\nMy Answer: (A or B) is more plausible.\" Remember that you should choose only one option for each answer."

    pattern = re.compile('\(?(A|B|yes|no|Yes|No)\)?')
    candidate, summary = [], []
    count = 0
    
    origin_hit, summary_hit, gpt4_hit = 0, 0, 0
    for c in tqdm(candidates):
        if c['Q'] in questions:
            continue
        
        debate = "\n".join([f"user{1 if i % 2 == 0 else 2}: {response['content']}" for i, response in enumerate(c['R'][2:])])
        if hps.dataset == "strategyqa":
            prompt = [{"role": "system", "content": instruction},
                      {"role": "user", "content": f"Question: {c['Q']}\n\nDebate:\n{debate}"}]
        else:
            prompt = [{"role": "system", "content": instruction},
                      {"role": "user", "content": f"Question: {c['Q']}\nOptions: {c['O']}\n\nDebate:\n{debate}"}]

        if origin_hit + summary_hit + gpt4_hit == 0:
            print(prompt)

        rs = get_response(hps, prompt)
        summary, gpt4_pred, summary_pred = get_prediction_explanation(hps.dataset, rs)
        answer = re.findall(pattern, c['A'])[0]

        c['S'] = summary

        if c['P'][-1] == c['P'][-2]:
            ori = c['P'][-1]
        else:
            ori = max(c['P'][2:], key=c['P'][2:].count)
        


        c['SP'] = {"Answer": answer, "Original": ori, "GPT4": gpt4_pred, "Summary": summary_pred}
        if ori == answer:
            origin_hit += 1
        if gpt4_pred == answer:
            gpt4_hit += 1
        if summary_pred == answer:
            summary_hit += 1

        fo.write(c)

    print('[Count]: {}'.format(len(candidates)))
    print('[Vote Hit]: {}'.format(origin_hit))
    print('[Summary Hit]: {}'.format(summary_hit))
    print('[GPT4 Hit]: {}'.format(gpt4_hit))
    fo.close()
