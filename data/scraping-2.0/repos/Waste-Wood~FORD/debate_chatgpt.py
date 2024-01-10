import pdb
import time
from argparse import ArgumentParser
from tools import read_jsonlines, define_logger
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
        # top_p=0.1
    )
    return responds


def get_prediction_explanation(task, response):
    text = response['choices'][0]['message']['content'].strip()
    # if task == 'csqa':
    #     try:
    #         explanation, prediction = text.split('Answer: ')
    #         prediction = re.findall(pattern, prediction.replace("Answer", ""))
    #         if len(prediction) >= 1:
    #             prediction = prediction[0]
    #         else:
    #             prediction = "None"
    #         explanation = explanation.strip().replace("Explanation: ", "")
    #     except:
    #         prediction = "None"
    #         explanation = text
    # else:
    try:
        text = text.split('Answer:')[1]
        prediction, explanation = text.split('Explanation: ')
        predictions = re.findall(pattern, prediction.replace("Answer", 'answer'))
        if len(predictions) >= 1:
            prediction = predictions[0]
        else:
            prediction = "None"
    except:
        explanation = text.strip()
        prediction = "None"
    if task == 'strategyqa':
        prediction = prediction.lower()
    return prediction, explanation


def hyper_parameters():
    parser = ArgumentParser('Few-Shot')

    parser.add_argument('--candidate', type=str, default='./output/debate/round0')
    parser.add_argument('--dataset', type=str, default='strategyqa')
    parser.add_argument('--api', type=str, default='')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--mode', type=str, default='debate')

    parser.add_argument('--log_dir', type=str, default='./output/logger')
    parser.add_argument('--log_name', type=str, default='')

    parser.add_argument('--output_dir', type=str, default='./output/debate/round1_chatgpt')
    parser.add_argument('--num_sequences', type=int, default=1)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--run', type=int, default=2)

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    hps = hyper_parameters()
    hps.log_name = '{}_{}.txt'.format(hps.mode, hps.dataset)

    logger = define_logger(hps)

    logger.info('[HPS]: {}'.format(hps))
    logger.info('[API Key]: {}'.format(hps.api))
    logger.info('[Data]: {}'.format(hps.dataset))
    logger.info('[Model]: {}'.format(hps.model))

    try:
        candidates = jsonlines.open(os.path.join(hps.candidate, '{}_candidate{}.jsonl'.format(hps.dataset, hps.run)), 'r')
    except:
        candidates = jsonlines.open(os.path.join(hps.candidate, '{}_candidate.jsonl'.format(hps.dataset)), 'r')
    candidates = [c for c in candidates]

    openai.api_key = hps.api
    if not os.path.exists(os.path.join(hps.output_dir)):
        os.makedirs(os.path.join(hps.output_dir))

    questions = []
    if os.path.exists(os.path.join(hps.output_dir, '{}_agreed{}.jsonl'.format(hps.dataset, hps.run))):
        fi = jsonlines.open(os.path.join(hps.output_dir, '{}_agreed{}.jsonl'.format(hps.dataset, hps.run)), 'r')
        fi2 = jsonlines.open(os.path.join(hps.output_dir, '{}_candidate{}.jsonl'.format(hps.dataset, hps.run)), 'r')
        questions += [d['Q'] for d in fi]
        questions += [d['Q'] for d in fi2]
        fi.close()
        fi2.close()
        fo1 = jsonlines.open(os.path.join(hps.output_dir, '{}_agreed{}.jsonl'.format(hps.dataset, hps.run)), 'a')
        fo2 = jsonlines.open(os.path.join(hps.output_dir, '{}_candidate{}.jsonl'.format(hps.dataset, hps.run)), 'a')
    else:
        fo1 = jsonlines.open(os.path.join(hps.output_dir, '{}_agreed{}.jsonl'.format(hps.dataset, hps.run)), 'w')
        fo2 = jsonlines.open(os.path.join(hps.output_dir, '{}_candidate{}.jsonl'.format(hps.dataset, hps.run)), 'w')
    
    # instruction = "Do you think I am more reasonable? Give your final answer starting with \"Answer: (A or B) is more plausible.\" and explain very shortly starting with \"Explanation: \". You should choose only one answer."
    
    if hps.dataset == 'csqa':
        instruction = "Do you think I am more reasonable or you have a different answer? Please give your final answer starting with \"Answer: (A|B|C|D|E) is more plausible.\" and explain very shortly starting with \"Explanation: \". You should choose only one answer."
    elif hps.dataset == 'socialiqa':
        instruction = "Do you think I am more reasonable? Please give your final answer starting with \"Answer: (A|B|C) is more plausible.\" and explain very shortly starting with \"Explanation: \". You should choose only one answer."
    elif hps.dataset == 'strategyqa':
        # instruction = "Please provide your opinion on whether my argument is more reasonable. Begin your response with \"Answer: (yes|no) is more plausible.\" and then briefly explain your reasoning, starting with \"Explanation: \". Remember, your answer should be either \"yes\" or \"no\"."
        instruction = "Read the question and our opinions carefully to get the key points of the question then consider the context to determine if the statement is really true or false. The question is hypothetical, do not introduce any new situation into the question. Use your general knowledge and understanding to provide your response with \"Answer: (yes|no) is more plausible.\" and then briefly explain your reasoning, starting with \"Explanation: \". Remember, your answer should be either \"yes\" or \"no\"."
    else:
        instruction = "Do you think I am more reasonable? Please give your final answer starting with \"Answer: (A|B) is more plausible.\" and explain very shortly starting with \"Explanation: \". You should choose only one answer."

    pattern = re.compile('\(?(A|B|C|D|E|F|G|yes|no|Yes|No)\)?')
    candidate, summary = [], []
    count = 0
    wr, rw, ww, rr = 0, 0, 0, 0
    for c in tqdm(candidates):
        if c['Q'] in questions:
            continue

        rounds = (len(c['R']) - 2) // 2 * 2
        debate_process = c['R'][:2] + [d for d in c['R'][-rounds:]]
        
        debate_process[-1]['content'] += f"\n{instruction}"
        # debate_process[-2]['content'] = f"The answer is ({c['P'][-2]})\n" + debate_process[-2]['content']
        debate_process[1]['content'] = f"{debate_process[1]['content']}\nPlease answer yes or no and give a short explanation with the format like 'Answer: _.\nExplanation: _'."
        
        if hps.model == "gpt-3.5-turbo-0301":
            for i in range(len(debate_process)):
                debate_process[i]['role'] = 'assistant' if debate_process[i]['role'] == 'user' else "user"

        prompt = debate_process
        if wr + rw + ww + rr == 0:
            print(prompt)

        rs = get_response(hps, prompt)
        prediction, explanation = get_prediction_explanation(hps.dataset, rs)
        debate_process[-1]['content'] = debate_process[-1]['content'].replace(f"\n{instruction}", "")
        debate_process[1]['content'] = debate_process[1]['content'].replace("\nPlease answer yes or no and give a short explanation with the format like 'Answer: _.\nExplanation: _'.", "")
        c['R'].append({"role": "assistant", "content": explanation.strip()})
        c['P'].append(prediction)
        c['M'].append('chatgpt' if hps.model == "gpt-3.5-turbo" else 'chatgpt0301')

        if hps.model == "gpt-3.5-turbo-0301":
            for i in range(len(debate_process)):
                debate_process[i]['role'] = 'assistant' if debate_process[i]['role'] == 'user' else "user"

        answer = re.findall(pattern, c['A'])[0]
        if prediction == c['P'][-2] == answer:
            summary.append(c)
            fo1.write(c)
            count += 1
            wr += 1
        elif prediction == c['P'][-2]:
            summary.append(c)
            fo1.write(c)
            rw += 1
        elif prediction == c['P'][-3] == answer:
            rr += 1
            candidate.append(c)
            fo2.write(c)
        else:
            candidate.append(c)
            fo2.write(c)
            ww += 1

    print('[Count]: {}'.format(count))
    print('[Summary]: {}'.format(len(summary)))
    print('[Candidate]: {}'.format(len(candidate)))
    print(f'[RW-WR-RR-WW]: {rw}-{wr}-{rr}-{ww}')
    fo1.close()
    fo2.close()
