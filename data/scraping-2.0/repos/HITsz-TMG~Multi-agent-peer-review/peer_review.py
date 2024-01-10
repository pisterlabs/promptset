# -*- coding: UTF-8 -*-
'''
=================================================
@Author : TMG HITSZ
@Date   : 2023/10/28
@Desc   : our proposed method
=================================================
'''

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from data_proc import check_dirs_files
import time
import json
import openai
from tqdm import tqdm
import jsonlines
import params



def read_json(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(output_path, output_data):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False)


def construct_assistant_message(completion):
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}


def generate_answer(answer_context):
    while True:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",  # 0301
                messages=answer_context,
                n=1)
            break
        except Exception as e:
            logging.warning(f"retrying due to an error: {e}")
            time.sleep(20)
    return completion


def read_jsonl(path: str):
    with open(path, encoding='utf-8') as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def peer_review(args):
    if not args.reload_data:
        generated_description = []
    else:
        with open(args.output_file, 'r', encoding='utf-8') as f:
            generated_description = json.load(f)
    generated_len = len(generated_description)
    if generated_len:
        logging.info(f'reload from: {args.output_file}')
        logging.info(f'reload data num: {generated_len}')

    all_datas = read_jsonl(args.task_file)
    for i, data in enumerate(tqdm(all_datas)):

        if args.reload_data and i < generated_len:
            continue

        question = data['question']
        answer = data['answer']

        if args.task in ['GSM8K', 'SVAMP', 'AddSub', 'SingleEq', 'MultiArith']:  # number
            content = """Can you solve the following math problem? {} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. """.format(question)
        elif args.task in ['AQuA', 'ARC-c', 'Colored_Objects', 'Penguins']:  # option
            content = "Can you answer the following question as accurately as possible? {} Explain your answer, putting the answer in the form (X) at the end of your response.".format(question)
        elif args.task in ['StrategyQA', ]:  # yes or no
            content = "Can you answer the following question as accurately as possible? {} Explain your answer, your answer should be Yes or No at the end of your response.".format(question)
        else:
            raise Exception('failed to construct question, unknown task!')
        agent_contexts = [[{"role": "user", "content": content}] for _ in range(args.agent_num)]

        agent_init_ans = None # agent initial answer
        agent_feedbacks = [[] for _ in range(args.agent_num)] # agent feedback
        for round_num in range(args.rounds):

            if  round_num == 1: # update agent initial answer
                agent_init_ans = [agent_contexts[k][1]['content'] for k in range(args.agent_num)]

            for j, agent_context in enumerate(agent_contexts):
                if round_num == 0: # ROUND 0: generate initial answer
                    completion = generate_answer(agent_context)
                    assistant_message = construct_assistant_message(completion)
                    agent_context.append(assistant_message)
                elif round_num == 1: # ROUND 1: give feedback to each other
                    ans_to_add = [k for k in range(args.agent_num) if k != j]
                    for index in ans_to_add:
                        init_ans = agent_init_ans[index]
                        # content = f"Here is a solution from another agent: \n\n {init_ans}\n\n Please examine this agent's reasoning process step by step and offer feedback on its reasoning."
                        content = f"Here is a solution from another agent: \n\n {init_ans}\n\n Please examine this agent's reasoning process step by step and offer feedback on its reasoning. " \
                                  f"You can rate your confidence in your feedback on a scale from 1-10, where 10 indicates the highest level of confidence."
                        agent_context.append({"role": "user", "content": content})
                        completion = generate_answer(agent_context)
                        assistant_message = construct_assistant_message(completion)
                        agent_context.append(assistant_message)
                        agent_feedbacks[index].append(assistant_message)
                elif round_num == 2: # ROUND 2: base on the initial answer and other angents' feedback, update final answer
                    agent_feedback = agent_feedbacks[j]
                    agent_num_dict = {1: "one", 2: "two", 3: "three", 4: "four"}
                    content = f"Here are the feedbacks for your solution from the above {agent_num_dict[args.agent_num - 1]} agents:\n\n "
                    for feedback in agent_feedback:
                        content += f"One agent feedback: {feedback['content']} \n\n "

                    if args.task in ['GSM8K', 'SVAMP', 'AddSub', 'SingleEq', 'MultiArith']:  # number
                        content += f"Using other agents' solutions and feedbacks as additional information, " \
                                   f"can you provide your answer to the math problem? \n " \
                                   f"The original math problem is {question}. " \
                                   f"Your final answer should be a single numerical number, " \
                                   f"in the form \\boxed{{answer}}, at the end of your response."
                    elif args.task in ['AQuA', 'ARC-c', 'Colored_Objects', 'Penguins']:  # option
                        content += f"Using the reasoning from other agents as additional advice, " \
                                   f"can you give an updated answer? Examine your solution and other agents' feedback step by step. " \
                                   f"Put your answer in the form (X) at the end of your response."
                    elif args.task in ['StrategyQA', ]:  # yes or no
                        content += f"Using the reasoning from other agents as additional advice, " \
                                   f"can you give an updated answer? Examine your solution and other agents' feedback step by step. " \
                                   f"Your answer should be Yes or No at the end of your response."
                    else:
                        raise Exception('failed to construct question, unknown task!')

                    # content_3 = f"Here are the feedbacks for your solution from the above two agents:\n\n One agent feedback: {agent_feedback[0]['content']} \n\n One agent feedback: {agent_feedback[1]['content']}\n\n Using other agents' solutions and feedbacks as additional information, can you provide your answer to the math problem? \n The original math problem is {question}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."
                    agent_context.append({"role": "user", "content": content})
                    completion = generate_answer(agent_context)
                    assistant_message = construct_assistant_message(completion)
                    agent_context.append(assistant_message)

        generated_description.append({
            'question': question,
            'answer': answer,
            'agent_contexts': agent_contexts,
        })
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(generated_description, f, ensure_ascii=False)


def log_param(args):
    args_str = f'\n---------------- peer review parameters ----------------\n'
    for k, v in args.__dict__.items():
        args_str += f'{k} = {v}\n'
    args_str += f'-------------------------------------------------------'
    logging.info(args_str)


if __name__ == "__main__":
    # 1. args
    args = params.peer_review_args()
    log_param(args)

    # 2. check dir and file
    check_dirs_files(dirs=[args.dataset_dir, args.output_dir, ], files=[args.task_file, ])

    # 3. key and org
    openai.api_key = args.openai_key
    openai.organization = args.openai_organization

    # 4. peer review method
    peer_review(args)