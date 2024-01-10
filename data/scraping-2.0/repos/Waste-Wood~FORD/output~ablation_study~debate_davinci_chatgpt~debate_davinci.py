import pdb
import time
from argparse import ArgumentParser
import jsonlines
from tqdm import tqdm
import os
import openai
import backoff
import re
import logging


def define_logger(hps):
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
    logging.basicConfig(format='%(asctime)s %(levelname)-8s: %(message)s', level=logging.INFO)
    logger = logging.getLogger(hps.log_name)

    # console_handler = logging.StreamHandler()
    # console_handler.formatter = formatter
    # console_handler.setLevel(logging.INFO)
    # logger.addHandler(console_handler)

    file_path = os.path.join(hps.log_dir, hps.log_name)
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger


@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.Timeout))
def get_response(hps, pmp):
    responds = openai.Completion.create(
        model=hps.model,
        prompt=pmp,
        max_tokens=hps.max_tokens,
        temperature=hps.temperature,
        api_key=openai.api_key,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return responds


def get_prediction_explanation(task, response):
    text = response['choices'][0]['text'].strip()
    try:
        prediction, explanation = text.split('Explanation: ')
        predictions = re.findall(pattern, prediction.replace("Answer", 'answer'))
        if len(prediction) >= 1:
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

    parser.add_argument('--candidate', type=str, default='./output/debate/round1_chatgpt/')
    parser.add_argument('--dataset', type=str, default='anli')
    parser.add_argument('--api', type=str, default='')
    parser.add_argument('--model', type=str, default='text-davinci-003')
    parser.add_argument('--mode', type=str, default='round2_davinci')

    parser.add_argument('--log_dir', type=str, default='./output/logger')
    parser.add_argument('--log_name', type=str, default='')

    parser.add_argument('--output_dir', type=str, default='./output/debate/round2_davinci')
    parser.add_argument('--num_sequences', type=int, default=1)
    parser.add_argument('--max_tokens', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.0)

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

    candidates = jsonlines.open(os.path.join(hps.candidate, '{}_candidate.jsonl'.format(hps.dataset)), 'r')
    candidates = [c for c in candidates]

    openai.api_key = hps.api
    if not os.path.exists(os.path.join(hps.output_dir)):
        os.makedirs(os.path.join(hps.output_dir))

    questions = []
    if os.path.exists(os.path.join(hps.output_dir, '{}_agreed.jsonl'.format(hps.dataset))):
        fi = jsonlines.open(os.path.join(hps.output_dir, '{}_agreed.jsonl'.format(hps.dataset)), 'r')
        fi2 = jsonlines.open(os.path.join(hps.output_dir, '{}_candidate.jsonl'.format(hps.dataset)), 'r')
        questions += [d['Q'] for d in fi]
        questions += [d['Q'] for d in fi2]
        fi.close()
        fi2.close()
        fo1 = jsonlines.open(os.path.join(hps.output_dir, '{}_agreed.jsonl'.format(hps.dataset)), 'a')
        fo2 = jsonlines.open(os.path.join(hps.output_dir, '{}_candidate.jsonl'.format(hps.dataset)), 'a')
    else:
        fo1 = jsonlines.open(os.path.join(hps.output_dir, '{}_agreed.jsonl'.format(hps.dataset)), 'w')
        fo2 = jsonlines.open(os.path.join(hps.output_dir, '{}_candidate.jsonl'.format(hps.dataset)), 'w')
    
    system = "You are in a debate now. My opinion is not always true, you can ignore any incorrect part of my opinion. And you can refer to my opinion to revise your choice or defend your own. Use your general knowledge and understanding to read my opinion carefully and compare it with your opinion based on the question. Please remember there should and must be a more plausible answer in the choices."
    if hps.dataset == "csqa":
        debate_instruction = "Read our opinions carefully to consider the question to determine the final answer choice. Do you think Me is more reasonable or you have a different answer? Please give your final answer starting with \"Answer: (A|B|C|D|E) is more plausible.\" and explain very shortly starting with \"Explanation: \". You should choose only one answer. Remember, our opinions might be all wrong."
        # debate_instruction = "Read our opinions carefully to consider the question to determine the final answer choice. Give your final answer starting with \"Answer: (A|B|C|D|E) is more plausible.\" and explain very shortly starting with \"Explanation: \". Your answer should be either \"A\", \"B\", \"C\", \"D\", or \"E\". Remember, our opinions might be all wrong."
    elif hps.dataset == 'strategyqa':
        debate_instruction = "Read the question and our opinions carefully to get the key points of the question then consider the context to determine if the statement is really true or false. The question is hypothetical, do not introduce any new situation into the question. Use your general knowledge and understanding to provide your response with \"Answer: (yes|no) is more plausible.\" and then briefly explain your reasoning, starting with \"Explanation: \". Remember, your answer should be either \"yes\" or \"no\"."
        # debate_instruction = "Read my opinion carefully and compare it with your opinion to consider the context to determine if the statement is really true or false. Use your general knowledge and understanding to provide your response with \"Answer: (yes|no) is more plausible.\" and then briefly explain your reasoning, starting with \"Explanation: \". Remember, your answer should be either \"yes\" or \"no\"."
    elif hps.dataset == 'socialiqa':
        debate_instruction = "Do you think I am more reasonable? Give your final answer starting with \"Answer: (A|B|C) is more plausible.\" and explain very shortly starting with \"Explanation: \". Your answer should be either \"A\", \"B\", or \"C\"."
    else:
        debate_instruction = "Do you think I am more reasonable? Give your final answer starting with \"Answer: (A|B) is more plausible.\" and explain very shortly starting with \"Explanation: \". Your answer should be either \"A\" or \"B\"."
    
    pattern = re.compile('\(?(A|B|C|D|E|Yes|No|yes|no)\)?')
    candidate, summary = [], []
    count = 0
    wr, rw, ww, rr = 0, 0, 0, 0
    for c in tqdm(candidates):

        if c['Q'] in questions:
            continue

        if hps.dataset == 'strategyqa':
            c['O'] = "(A) yes (B) no."

        rounds = (len(c['R']) - 2) // 2 * 2
        role1 = "You"
        role2 = "Me"
        debate_process = "\n".join([f"{role1 if i%2==0 else role2}: {d['content']}" for i, d in enumerate(c['R'][-rounds:])])
        if hps.dataset == 'strategyqa':
            prompt = f"{system}\n\nQuestion: {c['Q']}\n{debate_process}\n{debate_instruction}\nYou:"
        else:    
            prompt = f"{system}\n\nQuestion: {c['Q']}\nChoices: {c['O']}\n{debate_process}\n{debate_instruction}\nYou:"
        if ww + rw + rr + wr == 0:
            print(prompt)
        rs = get_response(hps, prompt)
        
        prediction, explanation = get_prediction_explanation(hps.dataset, rs)

        c['R'].append({"role": "user", "content": explanation})
        c['P'].append(prediction)
        c['M'].append('davinci')
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
