# -*- coding: UTF-8 -*-
'''
debate method (https://composable-models.github.io/llm_debate/)
=================================================
'''
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from data_proc import check_dirs_files
import openai
import json
from tqdm import tqdm
import params




def construct_message(agents, question, idx, task):
    if len(agents) == 0:
        return {"role": "user",
                "content": f"Can you double check that your answer is correct. Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{{answer}}."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    if task in ['GSM8K', 'SVAMP', 'AddSub', 'SingleEq', 'MultiArith']:  # number
        prefix_string = prefix_string + """\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n The original math problem is {}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.""".format(question)
    elif task in ['AQuA', 'ARC-c', 'Colored_Objects', 'Penguins']:  # option
        prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response.""".format(question)
    elif args.task in ['StrategyQA', ]: # yes or no
        prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Your answer should be Yes or No at the end of your response.""".format(question)
    else:
        raise Exception('failed to construct question, unknown task!')

    return {"role": "user", "content": prefix_string}


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


def debate(args):
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
        elif args.task in ['StrategyQA', ]: # yes or no
            content = "Can you answer the following question as accurately as possible? {} Explain your answer, your answer should be Yes or No at the end of your response.".format(question)
        else:
            raise Exception('failed to construct question, unknown task!')
        agent_contexts = [[{"role": "user", "content": content}] for _ in range(args.agent_num)]

        for round in range(args.rounds):
            for j, agent_context in enumerate(agent_contexts):

                if round != 0:
                    agent_contexts_other = agent_contexts[: j] + agent_contexts[j + 1:]
                    message = construct_message(agent_contexts_other, question, 2 * round - 1, args.task)
                    agent_context.append(message)

                # round == 0 or round != 0
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
    args_str = f'\n--------------- debate method parameters ---------------\n'
    for k, v in args.__dict__.items():
        args_str += f'{k} = {v}\n'
    args_str += f'-------------------------------------------------------'
    logging.info(args_str)


if __name__ == "__main__":
    # 1. args
    args = params.debate_args()
    log_param(args)

    # 2. check dir and file
    check_dirs_files(dirs=[args.dataset_dir, args.output_dir, ], files=[args.task_file, ])

    # 3. key and org
    openai.api_key = args.openai_key
    openai.organization = args.openai_organization

    # 4. debate method
    debate(args)






